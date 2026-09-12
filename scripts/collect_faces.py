"""
scripts/collect_faces.py
-------------------------
Command-line alternative to the web registration page
(/students/register). Registers a student in the database AND collects
~20-30 face images into dataset/students/<ID>_<Name>/.

Usage:
    python scripts/collect_faces.py

You will be prompted for Student ID, Name, Department and Email, then a
webcam preview window opens. Press SPACE to capture a validated face
image, 'q' to quit early.

Every candidate frame is validated with MTCNN before being saved -- if
no face is detected, or the crop is invalid, the frame is rejected and
you are told why, matching the same validation used by the web UI.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from app import config, database
from app.face_detector import FaceDetector
from app.utils import get_logger, sanitize_student_id, sanitize_name

logger = get_logger(__name__)


def prompt_student_details():
    print("\n=== Student Registration ===")
    student_id = sanitize_student_id(input("Student ID (e.g. ST001): "))
    name = sanitize_name(input("Full Name: "))
    department = input("Department: ").strip()
    email = input("Email: ").strip()
    return student_id, name, department, email


def main() -> int:
    database.init_db()

    student_id, name, department, email = prompt_student_details()
    if not student_id or not name:
        logger.error("Student ID and Name are required. Aborting.")
        return 1

    if database.student_exists(student_id):
        logger.error("Student ID %s already exists in the database.", student_id)
        return 1

    folder_name = f"{student_id}_{name}"
    folder_path = os.path.join(config.DATASET_PATH, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    if not database.add_student(student_id, name.replace("_", " "), department, email):
        logger.error("Failed to save student to the database.")
        return 1

    logger.info("Student %s (%s) added to the database.", name, student_id)
    logger.info("Face images will be saved to: %s", folder_path)

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        logger.error("Could not open camera index %s.", config.CAMERA_INDEX)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_FRAME_HEIGHT)

    detector = FaceDetector()
    target = config.NUM_COLLECTION_IMAGES
    captured = 0
    last_capture_time = 0.0

    print(f"\nLook at the camera. Press SPACE to capture ({target} images needed).")
    print("Vary your head angle/expression slightly between captures. Press 'q' to stop early.\n")

    try:
        while captured < target:
            ok, frame = cap.read()
            if not ok:
                logger.error("Failed to read frame from camera.")
                break

            display = frame.copy()
            face = detector.detect_largest(frame)

            if face is not None:
                x, y, w, h = face.box
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 200, 0), 2)
            else:
                cv2.putText(display, "No face detected", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(display, f"Captured: {captured}/{target}  SPACE=capture  q=quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("collect_faces.py", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key == ord(" "):
                now = time.time() * 1000
                if now - last_capture_time < config.COLLECTION_CAPTURE_DELAY_MS:
                    continue
                if face is None:
                    print("  [skip] No face detected, try again.")
                    continue
                crop = detector.crop_face(frame, face)
                if crop is None:
                    print("  [skip] Invalid crop, reposition and try again.")
                    continue
                crop_resized = cv2.resize(crop, config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                filename = os.path.join(folder_path, f"{captured + 1:03d}.jpg")
                cv2.imwrite(filename, crop_resized)
                captured += 1
                last_capture_time = now
                print(f"  [ok] Saved {filename} ({captured}/{target})")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if captured == 0:
        logger.warning("No images were captured. You can re-run this script to add images later.")
    else:
        logger.info("Captured %d/%d images for %s (%s).", captured, target, name, student_id)
        logger.info("Next steps: python scripts/extract_features.py  then  python scripts/train_model.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
