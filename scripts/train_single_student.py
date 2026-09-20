"""Create a fast recognizer for a dataset containing one student."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

from app import config
from app.face_recognizer import SingleClassClassifier
from app.utils import get_logger

logger = get_logger(__name__)
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def find_student_ids():
    student_ids = set()
    if not os.path.isdir(config.DATASET_PATH):
        return student_ids

    for folder_name in os.listdir(config.DATASET_PATH):
        folder_path = os.path.join(config.DATASET_PATH, folder_name)
        if not os.path.isdir(folder_path):
            continue
        has_images = any(
            filename.lower().endswith(VALID_EXTENSIONS)
            for filename in os.listdir(folder_path)
        )
        if has_images:
            student_ids.add(folder_name.split("_", 1)[0])
    return student_ids


def main():
    student_ids = find_student_ids()
    if len(student_ids) != 1:
        logger.error(
            "Fast single-student training requires exactly one student with images; found %d.",
            len(student_ids),
        )
        return 2

    student_id = next(iter(student_ids))
    label_encoder = LabelEncoder()
    label_encoder.fit([student_id])
    classifier = SingleClassClassifier()

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(classifier, config.SVM_MODEL_PATH)
    joblib.dump(label_encoder, config.LABEL_ENCODER_PATH)
    np.savez(
        os.path.join(config.MODEL_DIR, "test_split.npz"),
        X_test=np.empty((0, 0), dtype=np.float32),
        y_test=np.empty((0,), dtype=np.int64),
    )

    metadata = {
        "num_students": 1,
        "num_images": 0,
        "students": [student_id],
        "train_size": 0,
        "val_size": 0,
        "test_size": 0,
        "classifier_fit_size": 0,
        "validation_accuracy": None,
        "training_mode": "single_student_fast",
        "recognition_threshold": config.RECOGNITION_THRESHOLD,
    }
    with open(config.METADATA_PATH, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    logger.info("Created fast single-student model for %s.", student_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
