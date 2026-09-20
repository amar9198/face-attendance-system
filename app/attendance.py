"""
app/attendance.py
------------------
Business logic that sits between the recognizer and the database:

    Recognized student_id + confidence -> verify not already present today
                                         -> insert attendance record
                                         -> return a status the UI can show

This module is what "Attendance verification -> Database update" in the
architecture diagram refers to.
"""

from datetime import datetime
from typing import Dict, Any

from app import database
from app.utils import get_logger

logger = get_logger(__name__)


class AttendanceManager:
    """Wraps app.database with the specific duplicate-prevention rule
    used by the real-time recognition loop."""

    def __init__(self):
        # In-memory cache of student_ids already marked today, refreshed
        # from the DB at startup. This avoids one DB round-trip per video
        # frame for students who have already been marked, while the
        # UNIQUE(student_id, date) constraint in the database remains the
        # authoritative, race-condition-proof guard.
        self._today = datetime.now().strftime("%Y-%m-%d")
        self._marked_cache = set(database.get_today_present_ids(self._today))

    def _refresh_day_if_needed(self) -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._today:
            self._today = today
            self._marked_cache = set(database.get_today_present_ids(today))

    @staticmethod
    def get_student_name(student_id: str) -> str:
        student = database.get_student(student_id)
        return student["name"] if student else student_id

    def process_recognition(self, student_id: str, confidence: float) -> Dict[str, Any]:
        """Called once per recognized face per frame.

        Returns a dict describing what happened, e.g.:
            {"student_id": "ST001", "name": "John Doe", "status": "Present",
             "newly_marked": True, "confidence": 0.94}
        """
        self._refresh_day_if_needed()

        student = database.get_student(student_id)
        name = student["name"] if student else student_id

        if student_id in self._marked_cache:
            return {
                "student_id": student_id,
                "name": name,
                "status": "Already Present",
                "newly_marked": False,
                "confidence": confidence,
            }

        inserted = database.mark_attendance(student_id, confidence)
        if inserted:
            self._marked_cache.add(student_id)
            logger.info("Marked %s (%s) present at %.0f%% confidence", student_id, name, confidence * 100)
            return {
                "student_id": student_id,
                "name": name,
                "status": "Present",
                "newly_marked": True,
                "confidence": confidence,
            }
        else:
            # Lost a race with another frame/process between the cache
            # check and the insert -- the DB constraint caught it.
            self._marked_cache.add(student_id)
            return {
                "student_id": student_id,
                "name": name,
                "status": "Already Present",
                "newly_marked": False,
                "confidence": confidence,
            }
