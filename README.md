Classroom Attendance Demo

Project files:
- classroom_app.py: FastAPI backend and face enrollment, recognition, attendance, and reset endpoints.
- database.py: SQLite schema and student deletion helper.
- templates/index.html: browser dashboard with mirrored camera preview.
- requirements.txt: Python packages.

Setup on macOS:
1. Open Terminal and go to this project folder.
2. Create and activate a virtual environment:
   python3 -m venv .venv
   source .venv/bin/activate
3. Install packages:
   python -m pip install -r requirements.txt
4. Start the website:
   python -m uvicorn classroom_app:app --reload --port 8001
5. Open http://127.0.0.1:8001 in the browser and allow camera access.

Registration:
Enter a name and click Register. Follow the on-screen prompts for center, left, right, up, and down. The app stores a face template, not the camera image.

Attendance:
Click Start attendance. The first recognized appearance records In. If a registered face is absent for at least two seconds, its next recognized appearance toggles the visit to Out or starts another visit. Stop closes any currently open visits.

Reset and delete:
Reset time removes only that student's visit records; their name and face registration remain. Delete removes the student's registration and visit history.

The SQLite database file, classroom_attendance.db, is created beside classroom_app.py on first startup. Do not delete it unless you intend to erase all enrolled students and attendance records.

This is a classroom demonstration prototype. Use consenting volunteers and obtain appropriate school approval before collecting real student face data.
