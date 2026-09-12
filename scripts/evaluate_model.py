"""
scripts/evaluate_model.py
---------------------------
Evaluates the trained SVM on the TEST split that was set aside (and
saved to models/test_split.npz) by scripts/train_model.py -- i.e. data
the model never saw during training or validation.

Reports:
    Accuracy, Precision (macro), Recall (macro), F1-score (macro),
    a full per-class classification report, and a confusion matrix.

IMPORTANT: these numbers describe THIS system's performance on YOUR
dataset. They are not a claim about the reference paper's reported
accuracy -- the paper's numbers were produced on a different dataset
and are not automatically reproduced here.

Usage:
    python scripts/evaluate_model.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

from app import config
from app.utils import get_logger, ModelNotTrainedError

logger = get_logger(__name__)


def load_artifacts():
    required = [config.SVM_MODEL_PATH, config.LABEL_ENCODER_PATH]
    test_split_path = os.path.join(config.MODEL_DIR, "test_split.npz")
    required.append(test_split_path)

    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise ModelNotTrainedError(
            "Missing required file(s) for evaluation: "
            + ", ".join(missing)
            + ". Run scripts/extract_features.py then scripts/train_model.py first."
        )

    svm = joblib.load(config.SVM_MODEL_PATH)
    label_encoder = joblib.load(config.LABEL_ENCODER_PATH)
    split = np.load(test_split_path)
    return svm, label_encoder, split["X_test"], split["y_test"]


def main() -> int:
    try:
        svm, label_encoder, X_test, y_test = load_artifacts()
    except ModelNotTrainedError as exc:
        logger.error(str(exc))
        return 1

    if len(X_test) == 0:
        logger.error(
            "Test split is empty (likely too few images per student). "
            "Collect more face images per student and retrain."
        )
        return 1

    logger.info("Evaluating on %d held-out test samples...", len(X_test))

    y_pred = svm.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("\n===== Model Evaluation (held-out TEST set) =====")
    print(f"Test samples : {len(X_test)}")
    print(f"Accuracy     : {accuracy * 100:.2f}%")
    print(f"Precision    : {precision * 100:.2f}%  (macro-averaged)")
    print(f"Recall       : {recall * 100:.2f}%  (macro-averaged)")
    print(f"F1 Score     : {f1 * 100:.2f}%  (macro-averaged)")

    present_labels = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
    target_names = label_encoder.inverse_transform(present_labels)

    print("\n--- Per-class report ---")
    print(
        classification_report(
            y_test, y_pred,
            labels=present_labels,
            target_names=target_names,
            zero_division=0,
        )
    )

    print("--- Confusion matrix ---")
    cm = confusion_matrix(y_test, y_pred, labels=present_labels)
    header = "        " + " ".join(f"{n[:6]:>6}" for n in target_names)
    print(header)
    for name, row in zip(target_names, cm):
        print(f"{name[:6]:>6}  " + " ".join(f"{v:>6d}" for v in row))

    print(
        "\nNote: these figures reflect this system's own dataset and are "
        "NOT necessarily the same as any figures reported in the reference "
        "research paper.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
