"""
scripts/test_camera.py
-----------------------
Step 1 sanity check: Webcam -> MTCNN detection, with nothing else in the
pipeline. Run this FIRST after installing dependencies to confirm your
webcam and MTCNN are both working before moving on to registration.

Usage:
    python scripts/test_camera.py

Press 'q' to quit the preview window.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from app import config
from app.face_detector import FaceDetector
from app.utils import get_logger, CameraUnavailableError

logger = get_logger(__name__)


def main() -> int:
    logger.info("Opening camera at index %s ...", config.CAMERA_INDEX)
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        logger.error(
            "Could not open camera index %s. Check that a webcam is connected, "
            "not already in use, and that camera permissions are granted "
            "(see README Troubleshooting).",
            config.CAMERA_INDEX,
        )
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_FRAME_HEIGHT)

    detector = FaceDetector()
    logger.info("Camera opened. Loading MTCNN (first call is slower)...")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.error("Failed to read frame from camera.")
                break

            faces = detector.detect(frame)
            for face in faces:
                x, y, w, h = face.box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
                cv2.putText(
                    frame, f"{face.confidence*100:.1f}%", (x, max(y - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2,
                )

            cv2.putText(
                frame, f"Faces detected: {len(faces)}  (press 'q' to quit)",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
            )
            cv2.imshow("test_camera.py - MTCNN check", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera released.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
