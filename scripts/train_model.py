"""
scripts/train_model.py
------------------------
Trains the SVM classifier on the VGGFace embeddings produced by
scripts/extract_features.py:

    embeddings.npy + labels.npy
        -> train/val/test split
        -> SVM (scikit-learn, probability=True)
        -> models/svm_model.pkl
        -> models/label_encoder.pkl
        -> models/training_metadata.json (split sizes, val accuracy, classes)

The held-out validation split reported here is a quick sanity check.
scripts/evaluate_model.py performs the full accuracy/precision/recall/F1
evaluation on the separate TEST split.

Usage:
    python scripts/train_model.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from app import config
from app.utils import get_logger, EmptyDatasetError

logger = get_logger(__name__)


def load_features():
    if not (os.path.exists(config.EMBEDDINGS_PATH) and os.path.exists(config.LABELS_PATH)):
        raise EmptyDatasetError(
            "No extracted features found. Run 'python scripts/extract_features.py' first."
        )
    embeddings = np.load(config.EMBEDDINGS_PATH)
    labels = np.load(config.LABELS_PATH, allow_pickle=True)
    return embeddings, labels


def main() -> int:
    try:
        X, y = load_features()
    except EmptyDatasetError as exc:
        logger.error(str(exc))
        return 1

    classes, counts = np.unique(y, return_counts=True)
    logger.info("Loaded %d embeddings across %d students.", len(X), len(classes))
    for cls, cnt in zip(classes, counts):
        logger.info("  %s: %d images", cls, cnt)

    if len(classes) < 2:
        logger.error("Need at least 2 students with images to train a classifier.")
        return 1

    if np.min(counts) < 3:
        logger.warning(
            "At least one student has fewer than 3 images. Train/val/test "
            "splitting works best with >= 5 images per student; consider "
            "collecting more images with scripts/collect_faces.py."
        )

    # ---- Encode string student IDs -> integer class indices ---------------
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # ---- Train / validation / test split -----------------------------------
    # Stratify to keep class proportions consistent across splits wherever
    # the per-class count allows it (falls back to a non-stratified split
    # if a class has too few samples for stratification to be possible).
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y_encoded,
            test_size=(config.VAL_SPLIT + config.TEST_SPLIT),
            random_state=config.RANDOM_SEED,
            stratify=y_encoded,
        )
        relative_test_size = config.TEST_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=relative_test_size,
            random_state=config.RANDOM_SEED,
            stratify=y_temp,
        )
    except ValueError:
        logger.warning("Stratified split not possible (too few samples in a class); using a random split instead.")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y_encoded, test_size=(config.VAL_SPLIT + config.TEST_SPLIT), random_state=config.RANDOM_SEED,
        )
        relative_test_size = config.TEST_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=relative_test_size, random_state=config.RANDOM_SEED,
        )

    logger.info(
        "Split sizes -> train: %d, val: %d, test: %d",
        len(X_train), len(X_val), len(X_test),
    )

    # ---- Train the SVM -------------------------------------------------------
    logger.info(
        "Training SVM (kernel=%s, C=%s, probability=%s)...",
        config.SVM_KERNEL, config.SVM_C, config.SVM_PROBABILITY,
    )
    svm = SVC(
        kernel=config.SVM_KERNEL,
        C=config.SVM_C,
        probability=config.SVM_PROBABILITY,
        random_state=config.RANDOM_SEED,
    )
    svm.fit(X_train, y_train)

    # ---- Quick validation-set sanity check -----------------------------------
    val_preds = svm.predict(X_val) if len(X_val) else np.array([])
    val_accuracy = float(accuracy_score(y_val, val_preds)) if len(X_val) else None
    if val_accuracy is not None:
        logger.info("Validation accuracy: %.2f%%", val_accuracy * 100)
    else:
        logger.warning("Validation set was empty; skipping validation accuracy check.")

    # ---- Persist artifacts -----------------------------------------------------
    joblib.dump(svm, config.SVM_MODEL_PATH)
    joblib.dump(label_encoder, config.LABEL_ENCODER_PATH)

    # Save the test split indices/labels alongside so evaluate_model.py can
    # reproduce the SAME held-out test set deterministically.
    np.savez(
        os.path.join(config.MODEL_DIR, "test_split.npz"),
        X_test=X_test, y_test=y_test,
    )

    metadata = {
        "num_students": int(len(classes)),
        "num_images": int(len(X)),
        "students": list(map(str, classes)),
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "validation_accuracy": val_accuracy,
        "svm_kernel": config.SVM_KERNEL,
        "svm_C": config.SVM_C,
        "recognition_threshold": config.RECOGNITION_THRESHOLD,
    }
    with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved SVM model to %s", config.SVM_MODEL_PATH)
    logger.info("Saved label encoder to %s", config.LABEL_ENCODER_PATH)
    logger.info("Saved training metadata to %s", config.METADATA_PATH)
    logger.info("Next step: python scripts/evaluate_model.py  (or start the app: python app/main.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
