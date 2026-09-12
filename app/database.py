"""
app/database.py
----------------
SQLite data-access layer for the attendance system.

The database is created and migrated automatically the first time this
module is imported/used — the user never has to run manual SQL.

Tables
------
students:
    id            INTEGER PRIMARY KEY
    student_id    TEXT UNIQUE NOT NULL      (e.g. "ST001")
    name          TEXT NOT NULL
    department    TEXT
    email         TEXT
    created_at    TEXT NOT NULL

attendance:
    id            INTEGER PRIMARY KEY
    student_id    TEXT NOT NULL             (FK -> students.student_id)
    date          TEXT NOT NULL             (YYYY-MM-DD)
    time          TEXT NOT NULL             (HH:MM:SS)
    status        TEXT NOT NULL             ("Present")
    confidence    REAL NOT NULL

A UNIQUE(student_id, date) constraint on `attendance` enforces the
"one attendance record per student per day" rule at the database level,
in addition to the application-level check in app/attendance.py.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator, List, Optional, Dict, Any

from app import config
from app.utils import get_logger

logger = get_logger(__name__)


def get_connection() -> sqlite3.Connection:
    """Open a new SQLite connection with sane defaults."""
    conn = sqlite3.connect(config.DATABASE_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


@contextmanager
def db_cursor() -> Iterator[sqlite3.Cursor]:
    """Context manager that commits on success and rolls back on error."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create tables and indexes if they do not already exist.

    Called automatically on application startup (see app/main.py) and
    from every script that touches the database, so a missing database
    file never crashes the system.
    """
    with db_cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id    TEXT UNIQUE NOT NULL,
                name          TEXT NOT NULL,
                department    TEXT,
                email         TEXT,
                created_at    TEXT NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id    TEXT NOT NULL,
                date          TEXT NOT NULL,
                time          TEXT NOT NULL,
                status        TEXT NOT NULL DEFAULT 'Present',
                confidence    REAL NOT NULL,
                UNIQUE(student_id, date),
                FOREIGN KEY (student_id) REFERENCES students (student_id)
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_attendance_student ON attendance(student_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_students_student_id ON students(student_id);")
    logger.info("Database initialised at %s", config.DATABASE_PATH)


# ---------------------------------------------------------------------------
# Students
# ---------------------------------------------------------------------------
def add_student(student_id: str, name: str, department: str, email: str) -> bool:
    """Insert a new student. Returns False if the student_id already exists."""
    try:
        with db_cursor() as cur:
            cur.execute(
                "INSERT INTO students (student_id, name, department, email, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (student_id, name, department, email, datetime.now().isoformat(timespec="seconds")),
            )
        return True
    except sqlite3.IntegrityError:
        logger.warning("Student ID %s already exists.", student_id)
        return False


def get_student(student_id: str) -> Optional[Dict[str, Any]]:
    with db_cursor() as cur:
        cur.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def get_all_students() -> List[Dict[str, Any]]:
    with db_cursor() as cur:
        cur.execute("SELECT * FROM students ORDER BY created_at DESC")
        return [dict(r) for r in cur.fetchall()]


def student_exists(student_id: str) -> bool:
    return get_student(student_id) is not None


def delete_student(student_id: str) -> None:
    with db_cursor() as cur:
        cur.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
        cur.execute("DELETE FROM students WHERE student_id = ?", (student_id,))


# ---------------------------------------------------------------------------
# Attendance
# ---------------------------------------------------------------------------
def is_marked_today(student_id: str, date_str: Optional[str] = None) -> bool:
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    with db_cursor() as cur:
        cur.execute(
            "SELECT 1 FROM attendance WHERE student_id = ? AND date = ?",
            (student_id, date_str),
        )
        return cur.fetchone() is not None


def mark_attendance(student_id: str, confidence: float) -> bool:
    """Insert a new attendance record for *today*.

    Returns True if a new record was inserted, False if the student was
    already marked present today (no duplicate row is created). The
    UNIQUE(student_id, date) constraint guarantees this even under
    concurrent/rapid calls from consecutive video frames.
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    try:
        with db_cursor() as cur:
            cur.execute(
                "INSERT INTO attendance (student_id, date, time, status, confidence) "
                "VALUES (?, ?, ?, 'Present', ?)",
                (student_id, date_str, time_str, float(confidence)),
            )
        logger.info("Attendance marked: %s at %s", student_id, time_str)
        return True
    except sqlite3.IntegrityError:
        # Already marked today -- this is expected behaviour, not an error.
        return False


def get_attendance(
    date_str: Optional[str] = None,
    student_id: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch attendance records with optional filters, joined with student name."""
    query = (
        "SELECT a.id, a.student_id, s.name, s.department, a.date, a.time, "
        "a.status, a.confidence "
        "FROM attendance a LEFT JOIN students s ON a.student_id = s.student_id "
        "WHERE 1=1"
    )
    params: list = []
    if date_str:
        query += " AND a.date = ?"
        params.append(date_str)
    if student_id:
        query += " AND a.student_id = ?"
        params.append(student_id)
    if status:
        query += " AND a.status = ?"
        params.append(status)
    query += " ORDER BY a.date DESC, a.time DESC"

    with db_cursor() as cur:
        cur.execute(query, params)
        return [dict(r) for r in cur.fetchall()]


def get_today_present_ids(date_str: Optional[str] = None) -> List[str]:
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    with db_cursor() as cur:
        cur.execute("SELECT student_id FROM attendance WHERE date = ?", (date_str,))
        return [r["student_id"] for r in cur.fetchall()]


def get_dashboard_stats(date_str: Optional[str] = None) -> Dict[str, Any]:
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    total_students = len(get_all_students())
    present_today = len(get_today_present_ids(date_str))
    absent_today = max(total_students - present_today, 0)
    pct = round((present_today / total_students) * 100, 2) if total_students else 0.0
    return {
        "date": date_str,
        "total_students": total_students,
        "present_today": present_today,
        "absent_today": absent_today,
        "attendance_percentage": pct,
    }
