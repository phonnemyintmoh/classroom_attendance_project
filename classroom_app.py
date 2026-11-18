import json
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from insightface.app import FaceAnalysis
from pydantic import BaseModel

from database import DATABASE_FILE, delete_student, initialize_database


BASE_DIR = Path(__file__).resolve().parent
POSES = ("center", "left", "right", "up", "down")
TURN = 0.18
TILT = 0.08
APPEARANCE_GAP_SECONDS = 2.0
FACE_MATCH_THRESHOLD = 0.40

enrollment = {"name": None, "base": None, "shots": {}}
attendance = {"running": False, "last_seen": {}}


def yaw(face):
    left_eye, right_eye, nose = face.kps[0], face.kps[1], face.kps[2]
    eye_midpoint = (left_eye[0] + right_eye[0]) / 2
    eye_distance = max(right_eye[0] - left_eye[0], 1e-6)
    return (nose[0] - eye_midpoint) / eye_distance


def pitch(face):
    eyes_y = (face.kps[0][1] + face.kps[1][1]) / 2
    mouth_y = (face.kps[3][1] + face.kps[4][1]) / 2
    distance = max(mouth_y - eyes_y, 1e-6)
    return (face.kps[2][1] - eyes_y) / distance


def pose_of(yaw_change, pitch_change):
    if abs(yaw_change) >= TURN and abs(yaw_change) >= abs(pitch_change):
        return "left" if yaw_change > 0 else "right"
    if abs(pitch_change) >= TILT:
        return "down" if pitch_change > 0 else "up"
    if abs(yaw_change) < TURN / 2 and abs(pitch_change) < TILT / 2:
        return "center"
    return None


def quality_ok(frame, face):
    x1, y1, x2, y2 = face.bbox.astype(int)
    height, width = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)

    if min(x2 - x1, y2 - y1) < 80 or face.det_score < 0.6:
        return False

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = gray.mean()
    return sharpness >= 40 and 50 <= brightness <= 220


def pose_prompt(pose):
    prompts = {
        "center": "Look straight at the camera.",
        "left": "Slowly turn your head to your left.",
        "right": "Slowly turn your head to your right.",
        "up": "Slowly tilt your head upward.",
        "down": "Slowly tilt your head downward.",
    }
    return prompts[pose]


def next_pose():
    for pose in POSES:
        if pose not in enrollment["shots"]:
            return pose
    return None


def attendance_summary():
    now = datetime.now().astimezone()
    summary = []

    with sqlite3.connect(DATABASE_FILE) as connection:
        students = connection.execute(
            "SELECT id, name FROM students ORDER BY name"
        ).fetchall()

        for student_id, name in students:
            visits = connection.execute(
                "SELECT entry_time, exit_time FROM visits WHERE student_id = ?",
                (student_id,),
            ).fetchall()

            total_seconds = 0
            currently_in = False

            for entry_text, exit_text in visits:
                entry = datetime.fromisoformat(entry_text)
                end = datetime.fromisoformat(exit_text) if exit_text else now
                total_seconds += max(0, int((end - entry).total_seconds()))
                if exit_text is None:
                    currently_in = True

            hours, remainder = divmod(total_seconds, 3600)
            minutes = remainder // 60
            summary.append({
                "id": student_id,
                "name": name,
                "status": "In" if currently_in else "Out",
                "total_time": f"{hours}h {minutes}m",
            })

    return summary


def load_face_templates():
    records = []

    with sqlite3.connect(DATABASE_FILE) as connection:
        rows = connection.execute(
            """
            SELECT id, name, face_template
            FROM students
            WHERE face_template IS NOT NULL
            """
        ).fetchall()

    for student_id, name, template_text in rows:
        try:
            embedding = np.asarray(json.loads(template_text), dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm == 0:
                continue
            records.append((student_id, name, embedding / norm))
        except (ValueError, TypeError, json.JSONDecodeError):
            continue

    return records


def match_faces(faces, records):
    possibilities = []

    for face_index, face in enumerate(faces):
        face_embedding = face.normed_embedding.astype(np.float32)
        for record_index, (_, _, student_embedding) in enumerate(records):
            score = float(np.dot(face_embedding, student_embedding))
            if score >= FACE_MATCH_THRESHOLD:
                possibilities.append((score, face_index, record_index))

    possibilities.sort(reverse=True)
    used_faces = set()
    used_students = set()
    matches = []

    for score, face_index, record_index in possibilities:
        if face_index in used_faces or record_index in used_students:
            continue

        used_faces.add(face_index)
        used_students.add(record_index)
        student_id, name, _ = records[record_index]
        matches.append((student_id, name, score))

    return matches


@asynccontextmanager
async def lifespan(app):
    initialize_database()
    model = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
    )
    model.prepare(ctx_id=-1, det_size=(640, 640))
    app.state.face_model = model
    yield


app = FastAPI(title="Classroom Attendance", lifespan=lifespan)


class RegistrationRequest(BaseModel):
    name: str


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.post("/api/register/start")
def start_registration(body: RegistrationRequest):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Enter a student's name.")
    if enrollment["name"] is not None:
        raise HTTPException(status_code=409, detail="A registration is already running.")

    enrollment["name"] = name
    enrollment["base"] = None
    enrollment["shots"] = {}
    return {"started": True, "message": pose_prompt("center")}


@app.post("/api/register/cancel")
def cancel_registration():
    enrollment["name"] = None
    enrollment["base"] = None
    enrollment["shots"] = {}
    return {"cancelled": True}


@app.post("/api/register/frame")
async def receive_registration_frame(request: Request):
    if enrollment["name"] is None:
        raise HTTPException(status_code=409, detail="Click Register first.")

    raw_image = await request.body()
    frame = cv2.imdecode(np.frombuffer(raw_image, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not read camera frame.")

    faces = request.app.state.face_model.get(frame)
    if not faces:
        return {"message": "No face detected. Look toward the camera."}
    if len(faces) > 1:
        return {"message": "Register one person at a time; only one face should be visible."}

    face = faces[0]
    if not quality_ok(frame, face):
        return {"message": "Move closer and use brighter, steady lighting."}

    if enrollment["base"] is None:
        enrollment["base"] = (yaw(face), pitch(face))

    yaw_change = yaw(face) - enrollment["base"][0]
    pitch_change = pitch(face) - enrollment["base"][1]
    detected_pose = pose_of(yaw_change, pitch_change)
    wanted_pose = next_pose()

    if detected_pose == wanted_pose:
        enrollment["shots"][wanted_pose] = face.normed_embedding.astype(np.float32)
        following_pose = next_pose()

        if following_pose:
            return {
                "message": (
                    f"Captured {len(enrollment['shots'])}/{len(POSES)}. "
                    f"{pose_prompt(following_pose)}"
                )
            }

        template = np.mean(list(enrollment["shots"].values()), axis=0)
        template = template / np.linalg.norm(template)
        saved_name = enrollment["name"]
        created_at = datetime.now().astimezone().isoformat(timespec="seconds")

        with sqlite3.connect(DATABASE_FILE) as connection:
            connection.execute(
                """
                INSERT INTO students (name, face_template, created_at)
                VALUES (?, ?, ?)
                """,
                (saved_name, json.dumps(template.tolist()), created_at),
            )

        enrollment["name"] = None
        enrollment["base"] = None
        enrollment["shots"] = {}
        return {"done": True, "message": f"Registered {saved_name} successfully."}

    return {
        "message": (
            f"Captured {len(enrollment['shots'])}/{len(POSES)}. "
            f"{pose_prompt(wanted_pose)}"
        )
    }


@app.get("/api/students")
def get_students():
    with sqlite3.connect(DATABASE_FILE) as connection:
        rows = connection.execute(
            "SELECT id, name FROM students ORDER BY name"
        ).fetchall()
    return [{"id": row[0], "name": row[1]} for row in rows]


@app.delete("/api/students/{student_id}")
def remove_student(student_id: int):
    if not delete_student(student_id):
        raise HTTPException(status_code=404, detail="Student not found.")
    attendance["last_seen"].pop(student_id, None)
    return {"deleted": True}


@app.post("/api/students/{student_id}/reset-time")
def reset_student_time(student_id: int):
    with sqlite3.connect(DATABASE_FILE) as connection:
        exists = connection.execute(
            "SELECT id FROM students WHERE id = ?",
            (student_id,),
        ).fetchone()

        if exists is None:
            raise HTTPException(status_code=404, detail="Student not found.")

        connection.execute(
            "DELETE FROM visits WHERE student_id = ?",
            (student_id,),
        )

    attendance["last_seen"].pop(student_id, None)
    return {"reset": True}


@app.post("/api/attendance/start")
def start_attendance():
    now_text = datetime.now().astimezone().isoformat(timespec="seconds")
    with sqlite3.connect(DATABASE_FILE) as connection:
        connection.execute(
            "UPDATE visits SET exit_time = ? WHERE exit_time IS NULL",
            (now_text,),
        )

    attendance["last_seen"].clear()
    attendance["running"] = True
    return {"running": True, "message": "Attendance detection started."}


@app.post("/api/attendance/stop")
def stop_attendance():
    now_text = datetime.now().astimezone().isoformat(timespec="seconds")
    with sqlite3.connect(DATABASE_FILE) as connection:
        connection.execute(
            "UPDATE visits SET exit_time = ? WHERE exit_time IS NULL",
            (now_text,),
        )

    attendance["running"] = False
    attendance["last_seen"].clear()
    return {
        "running": False,
        "message": "Attendance stopped. Open visits were closed.",
        "students": attendance_summary(),
    }


@app.post("/api/attendance/frame")
async def receive_attendance_frame(request: Request):
    if not attendance["running"]:
        raise HTTPException(status_code=409, detail="Click Start Attendance first.")

    raw_image = await request.body()
    frame = cv2.imdecode(np.frombuffer(raw_image, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not read camera frame.")

    faces = request.app.state.face_model.get(frame)
    matches = match_faces(faces, load_face_templates())
    monotonic_now = time.monotonic()
    event_time = datetime.now().astimezone().isoformat(timespec="seconds")
    events = []

    with sqlite3.connect(DATABASE_FILE) as connection:
        for student_id, name, score in matches:
            last_seen = attendance["last_seen"].get(student_id)
            new_appearance = (
                last_seen is None
                or monotonic_now - last_seen >= APPEARANCE_GAP_SECONDS
            )

            if new_appearance:
                open_visit = connection.execute(
                    """
                    SELECT id FROM visits
                    WHERE student_id = ? AND exit_time IS NULL
                    ORDER BY id DESC LIMIT 1
                    """,
                    (student_id,),
                ).fetchone()

                if open_visit:
                    connection.execute(
                        "UPDATE visits SET exit_time = ? WHERE id = ?",
                        (event_time, open_visit[0]),
                    )
                    event = "Out"
                else:
                    connection.execute(
                        """
                        INSERT INTO visits (student_id, entry_time)
                        VALUES (?, ?)
                        """,
                        (student_id, event_time),
                    )
                    event = "In"

                events.append({
                    "name": name,
                    "event": event,
                    "score": round(score, 3),
                })

            attendance["last_seen"][student_id] = monotonic_now

    return {"events": events, "students": attendance_summary()}


@app.get("/api/attendance/state")
def get_attendance_state():
    return {
        "running": attendance["running"],
        "students": attendance_summary(),
    }
