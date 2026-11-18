import sqlite3
from pathlib import Path

DATABASE_FILE = Path(__file__).with_name("classroom_attendance.db")


def initialize_database():
    with sqlite3.connect(DATABASE_FILE) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                face_template TEXT,
                created_at TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
            )
        """)


def delete_student(student_id):
    with sqlite3.connect(DATABASE_FILE) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        exists = connection.execute(
            "SELECT id FROM students WHERE id = ?",
            (student_id,),
        ).fetchone()

        if exists is None:
            return False

        connection.execute(
            "DELETE FROM visits WHERE student_id = ?",
            (student_id,),
        )
        connection.execute(
            "DELETE FROM students WHERE id = ?",
            (student_id,),
        )

    return True
