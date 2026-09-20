"""
app/face_recognizer.py
-----------------------
Stage 2 (feature extraction) and stage 3 (classification) of the
pipeline:

    Face image/crop -> VGGFace embedding -> SVM -> student_id

FaceEmbedder is used by both:

    scripts/extract_features.py
    Real-time face recognition

This ensures the same feature extraction pipeline is used everywhere.
"""

import os
from typing import List, Optional, Tuple

import cv2
import joblib
import numpy as np

from app import config
from app.utils import (
    get_logger,
    ModelNotTrainedError,
    InvalidFaceCropError,
)

logger = get_logger(__name__)


class SingleClassClassifier:
    """Minimal probability classifier for datasets containing one student."""

    def __init__(self):
        self.classes_ = np.array([0], dtype=np.int64)

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=np.int64)

    def predict_proba(self, X):
        return np.ones((len(X), 1), dtype=np.float32)


# ============================================================
# FACE EMBEDDING
# ============================================================

class FaceEmbedder:
    """
    Extracts a fixed-length VGG-Face embedding vector.

    DeepFace is used as the backend for the VGG-Face model.
    """

    _deepface_module = None

    def __init__(self, model_name: str = config.VGGFACE_MODEL_NAME):
        self.model_name = model_name

    def _get_deepface(self):
        """
        Load TensorFlow/Keras first, then import DeepFace.

        This import order helps avoid tensorflow.keras compatibility
        problems with MTCNN/DeepFace.
        """

        if FaceEmbedder._deepface_module is None:

            try:
                # ------------------------------------------------
                # IMPORTANT: Initialize TensorFlow + Keras first
                # ------------------------------------------------
                import tensorflow as tf
                from tensorflow.keras.layers import Input

                logger.info(
                    "TensorFlow %s and tensorflow.keras initialized.",
                    tf.__version__
                )

                # ------------------------------------------------
                # Import DeepFace after TensorFlow/Keras
                # ------------------------------------------------
                from deepface import DeepFace

                logger.info("DeepFace module loaded successfully.")

                FaceEmbedder._deepface_module = DeepFace

            except Exception as exc:

                logger.exception(
                    "Failed to initialize DeepFace."
                )

                raise RuntimeError(
                    f"DeepFace initialization failed: {exc}"
                ) from exc

        return FaceEmbedder._deepface_module

    @staticmethod
    def _validate(face_bgr: np.ndarray) -> np.ndarray:
        """
        Validate the input face image.
        """

        if face_bgr is None:
            raise InvalidFaceCropError(
                "Cannot embed an empty face crop."
            )

        if face_bgr.size == 0:
            raise InvalidFaceCropError(
                "Cannot embed an empty face crop."
            )

        return face_bgr

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Extract a single VGG-Face embedding.

        Returns:
            numpy.ndarray:
                L2-normalized face embedding vector.
        """

        # Validate image
        face_bgr = self._validate(face_bgr)

        # Load DeepFace
        deepface = self._get_deepface()

        try:

            # Generate VGG-Face embedding
            result = deepface.represent(
                img_path=face_bgr,
                model_name=self.model_name,

                # The image is already treated as a face crop
                detector_backend="skip",

                enforce_detection=False,

                # No additional alignment
                align=False,
            )

        except Exception as exc:

            logger.exception(
                "Failed to generate face embedding."
            )

            raise RuntimeError(
                f"Face embedding generation failed: {exc}"
            ) from exc

        # Convert embedding to NumPy array
        embedding = np.array(
            result[0]["embedding"],
            dtype=np.float32
        )

        # --------------------------------------------------------
        # L2 NORMALIZATION
        # --------------------------------------------------------

        norm = np.linalg.norm(embedding)

        if norm > 0:
            embedding = embedding / norm

        return embedding

    def embed_batch(
        self,
        face_crops: List[np.ndarray]
    ) -> np.ndarray:
        """
        Extract embeddings for multiple face images.
        """

        embeddings = []

        for face in face_crops:
            embedding = self.embed(face)
            embeddings.append(embedding)

        return np.stack(
            embeddings,
            axis=0
        )


# ============================================================
# FACE RECOGNITION
# ============================================================

class FaceRecognizer:
    """
    Loads the trained SVM model and identifies faces.

    Pipeline:

        Face
          ↓
        VGG-Face Embedding
          ↓
        SVM Classifier
          ↓
        Student ID
    """

    def __init__(
        self,
        svm_path: str = config.SVM_MODEL_PATH,
        encoder_path: str = config.LABEL_ENCODER_PATH,
        recognition_threshold: float = config.RECOGNITION_THRESHOLD,
    ):

        self.svm_path = svm_path
        self.encoder_path = encoder_path

        self.recognition_threshold = recognition_threshold

        # Face embedding model
        self.embedder = FaceEmbedder()

        # Models are loaded lazily
        self._svm = None
        self._label_encoder = None

    @property
    def is_trained(self) -> bool:
        """
        Check whether trained model files exist.
        """

        return (
            os.path.exists(self.svm_path)
            and os.path.exists(self.encoder_path)
        )

    def _load(self):
        """
        Load trained SVM and LabelEncoder.
        """

        if (
            self._svm is None
            or self._label_encoder is None
        ):

            if not self.is_trained:

                raise ModelNotTrainedError(
                    "No trained model found.\n"
                    "Run these commands first:\n"
                    "1. python scripts/extract_features.py\n"
                    "2. python scripts/train_model.py"
                )

            logger.info(
                "Loading trained SVM model..."
            )

            self._svm = joblib.load(
                self.svm_path
            )

            self._label_encoder = joblib.load(
                self.encoder_path
            )

            logger.info(
                "SVM model and label encoder loaded successfully."
            )

    def identify(
        self,
        face_bgr: np.ndarray
    ) -> Tuple[str, Optional[str], float]:
        """
        Identify a face.

        Returns:

            status:
                "known" or "unknown"

            student_id:
                Student ID if recognized,
                otherwise None

            confidence:
                Recognition confidence between 0 and 1
        """

        # Load trained models
        self._load()

        # Generate face embedding
        embedding = self.embedder.embed(
            face_bgr
        )

        # Predict probabilities
        probabilities = self._svm.predict_proba(
            [embedding]
        )[0]

        # Find highest probability
        best_idx = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[best_idx]
        )

        # --------------------------------------------------------
        # UNKNOWN FACE CHECK
        # --------------------------------------------------------

        if confidence < self.recognition_threshold:

            logger.info(
                "Unknown face detected. Confidence: %.4f",
                confidence
            )

            return (
                "unknown",
                None,
                confidence
            )

        # Convert predicted class to Student ID
        student_id = self._label_encoder.inverse_transform(
            [best_idx]
        )[0]

        logger.info(
            "Face recognized: %s | Confidence: %.4f",
            student_id,
            confidence
        )

        return (
            "known",
            student_id,
            confidence
        )