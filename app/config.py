"""
app/config.py
-------------
Single source of truth for every configurable value in the project.

Nothing else in the codebase should hard-code camera indexes, thresholds,
image sizes or file paths — everything imports from here so behaviour can
be tuned in one place.

Values can be overridden with environment variables, which makes it easy
to run the same code in different environments (e.g. a lab PC with a
USB webcam at index 1) without touching source code.
"""

import os

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(BASE_DIR, "dataset", "students")
MODEL_DIR = os.path.join(BASE_DIR, "models")
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
EMBEDDINGS_PATH = os.path.join(MODEL_DIR, "embeddings.npy")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.npy")
METADATA_PATH = os.path.join(MODEL_DIR, "training_metadata.json")
EMBEDDING_CACHE_PATH = os.path.join(MODEL_DIR, "embedding_cache.json")

ATTENDANCE_DIR = os.path.join(BASE_DIR, "attendance")
DATABASE_PATH = os.path.join(ATTENDANCE_DIR, "attendance.db")

REPORTS_DIR = os.path.join(BASE_DIR, "reports")

LOG_DIR = os.path.join(BASE_DIR, "logs")

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", 0))

CAMERA_FRAME_WIDTH = int(
    os.environ.get("CAMERA_FRAME_WIDTH", 640)
)

CAMERA_FRAME_HEIGHT = int(
    os.environ.get("CAMERA_FRAME_HEIGHT", 480)
)
# ---------------------------------------------------------------------------
# Face detection (MTCNN)
# ---------------------------------------------------------------------------
# Minimum MTCNN confidence for a detection to be accepted at all.
FACE_CONFIDENCE_THRESHOLD = float(os.environ.get("FACE_CONFIDENCE_THRESHOLD", 0.75))

# Faces smaller than this (in pixels, on the smaller side of the bbox) are
# discarded as noise / false positives.
MIN_FACE_SIZE = int(os.environ.get("MIN_FACE_SIZE", 25))

# ---------------------------------------------------------------------------
# Feature extraction (VGGFace)
# ---------------------------------------------------------------------------
# Input size expected by the VGGFace network.
IMAGE_SIZE = (224, 224)

# Which VGGFace-compatible model backend to use for feature extraction.
# VGG-Face is accurate but very memory-heavy on typical laptops; use the
# lighter Facenet default here to avoid the ArrayMemoryError seen on local
# Windows machines.
VGGFACE_MODEL_NAME = os.environ.get("VGGFACE_MODEL_NAME", "Facenet")

# ---------------------------------------------------------------------------
# Recognition / classification (SVM)
# ---------------------------------------------------------------------------
# Minimum SVM probability required to show the predicted student's name.
RECOGNITION_THRESHOLD = float(os.environ.get("RECOGNITION_THRESHOLD", 0.40))

# A displayed identity is only allowed to create attendance at this higher
# confidence level.
ATTENDANCE_THRESHOLD = float(
    os.environ.get("ATTENDANCE_THRESHOLD", 0.50)
)

# Browser-camera liveness gate. Recognition requires an open -> closed ->
# open eye sequence before attendance is accepted.
LIVENESS_REQUIRED_BLINKS = int(
    os.environ.get("LIVENESS_REQUIRED_BLINKS", 1)
)

# SVM hyper-parameters (used by scripts/train_model.py)
SVM_KERNEL = "linear"
SVM_C = 1.0
SVM_PROBABILITY = True  # required to get class probabilities for thresholding

# Dataset split ratios (must sum to 1.0)
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Attendance rules
# ---------------------------------------------------------------------------
# A student is only marked present once per day. Set to False to allow a
# new "session" concept (e.g. per class period) if you extend the schema.
ONE_ATTENDANCE_PER_DAY = True

# ---------------------------------------------------------------------------
# Face collection (registration)
# ---------------------------------------------------------------------------
NUM_COLLECTION_IMAGES = int(os.environ.get("NUM_COLLECTION_IMAGES", 5))
MIN_COLLECTION_IMAGES = 5
MAX_COLLECTION_IMAGES = 30
COLLECTION_CAPTURE_DELAY_MS = 250  # minimum delay between accepted captures

# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------
FLASK_HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(
    os.environ.get(
        "PORT",
        os.environ.get("FLASK_PORT", 5000)
    )
)
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() == "true"

SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-secret-key")

# ---------------------------------------------------------------------------
# Registration email (SMTP)
# ---------------------------------------------------------------------------
SMTP_HOST = os.environ.get("SMTP_HOST", "").strip()
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "").strip()
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_FROM = os.environ.get("SMTP_FROM", "").strip()
SMTP_USE_TLS = os.environ.get("SMTP_USE_TLS", "true").lower() == "true"
SMTP_USE_SSL = os.environ.get("SMTP_USE_SSL", "false").lower() == "true"
SMTP_TIMEOUT = int(os.environ.get("SMTP_TIMEOUT", "20"))

# ---------------------------------------------------------------------------
# Ensure required directories exist (created automatically, no manual setup)
# ---------------------------------------------------------------------------
for _path in (DATASET_PATH, MODEL_DIR, ATTENDANCE_DIR, REPORTS_DIR, LOG_DIR):
    os.makedirs(_path, exist_ok=True)
