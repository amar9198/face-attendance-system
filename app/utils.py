"""
app/utils.py
------------
Small shared utilities: logging setup, image validation helpers, and
custom exceptions used across the pipeline so error handling is
consistent between the scripts and the Flask app.
"""

import logging
import os
import sys
from typing import Optional

import numpy as np

from app import config


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that writes to console and to a shared
    log file under logs/attendance_system.log."""
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured (avoid duplicate handlers on re-import)
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    try:
        log_path = os.path.join(config.LOG_DIR, "attendance_system.log")
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # If the filesystem is read-only or the path is unavailable, fall
        # back to console-only logging rather than crashing the app.
        logger.warning("Could not open log file, continuing with console logging only.")

    return logger


# ---------------------------------------------------------------------------
# Custom exceptions (make error handling explicit and specific)
# ---------------------------------------------------------------------------
class CameraUnavailableError(Exception):
    """Raised when the webcam cannot be opened or read from."""


class NoFaceDetectedError(Exception):
    """Raised when MTCNN finds no face in a frame that requires one."""


class InvalidFaceCropError(Exception):
    """Raised when a detected bounding box produces an unusable crop."""


class ModelNotTrainedError(Exception):
    """Raised when recognition is attempted before a model has been trained."""


class EmptyDatasetError(Exception):
    """Raised when training/feature extraction is attempted on an empty dataset."""


class CorruptedImageError(Exception):
    """Raised when an image file exists but cannot be decoded."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def is_valid_bbox(x: int, y: int, w: int, h: int, frame_shape) -> bool:
    """Validate that a bounding box is sane and fully inside the frame."""
    if w <= 0 or h <= 0:
        return False
    if x < 0 or y < 0:
        return False
    frame_h, frame_w = frame_shape[0], frame_shape[1]
    if x + w > frame_w or y + h > frame_h:
        return False
    if w < config.MIN_FACE_SIZE or h < config.MIN_FACE_SIZE:
        return False
    return True


def safe_crop(frame: np.ndarray, x: int, y: int, w: int, h: int) -> Optional[np.ndarray]:
    """Crop a region from a frame, clamping to image bounds. Returns None
    if the resulting crop is empty/invalid."""
    frame_h, frame_w = frame.shape[0], frame.shape[1]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame_w, x + w), min(frame_h, y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def sanitize_student_id(student_id: str) -> str:
    """Keep folder names filesystem-safe."""
    return "".join(c for c in student_id.strip() if c.isalnum() or c in ("-", "_")).upper()


def sanitize_name(name: str) -> str:
    return "".join(c for c in name.strip() if c.isalnum() or c in (" ", "-", "_")).replace(" ", "_")
