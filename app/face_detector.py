"""
app/face_detector.py
---------------------
MTCNN-based face detection.

This is stage 1 of the pipeline:

    Webcam frame -> MTCNN -> list of DetectedFace(bbox, confidence, landmarks)

Used by:
  - scripts/collect_faces.py   (registration image capture)
  - app/face_recognizer.py     (real-time recognition, via app/camera.py)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import cv2
import numpy as np

from app import config
from app.utils import get_logger, is_valid_bbox, safe_crop

logger = get_logger(__name__)


@dataclass
class DetectedFace:
    """One face detected in a frame."""
    box: Tuple[int, int, int, int]          # (x, y, w, h)
    confidence: float
    landmarks: Optional[Dict[str, Tuple[int, int]]] = field(default=None)

    @property
    def crop_coords(self) -> Tuple[int, int, int, int]:
        return self.box


class FaceDetector:
    """Thin, defensive wrapper around the `mtcnn` package.

    The underlying MTCNN model is loaded lazily on first use so that
    importing this module (e.g. for unit tests, or when only the SVM
    part of the pipeline is needed) does not require TensorFlow to be
    initialised immediately.
    """

    def __init__(
        self,
        confidence_threshold: float = config.FACE_CONFIDENCE_THRESHOLD,
        min_face_size: int = config.MIN_FACE_SIZE,
    ):
        self.confidence_threshold = confidence_threshold
        self.min_face_size = min_face_size
        self._detector = None  # lazy-loaded MTCNN instance

    def _load(self):
       if self._detector is None:
        try:
            from mtcnn import MTCNN
        except ImportError as exc:
            raise ImportError(
                "The 'mtcnn' package is not installed. Run: pip install mtcnn"
            ) from exc

        logger.info("Loading MTCNN face detector...")

        self._detector = MTCNN()

        logger.info("MTCNN loaded successfully.")

       return self._detector

    def detect(self, frame_bgr: np.ndarray) -> List[DetectedFace]:
        """Detect all faces in a BGR frame (as returned by OpenCV).

        Returns an empty list (never raises) if no faces are found, so
        callers can safely do `for face in detector.detect(frame):`.
        Detections below `confidence_threshold` or with an invalid/too
        small bounding box are silently filtered out.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            logger.warning("detect() called with an empty/invalid frame.")
            return []

        detector = self._load()

        # mtcnn expects RGB images
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        try:
            raw_results = detector.detect_faces(frame_rgb)
        except Exception as exc:  # noqa: BLE001 - defensive: never crash the app
            logger.error("MTCNN detection failed: %s", exc)
            return []

        faces: List[DetectedFace] = []
        for result in raw_results:
            confidence = float(result.get("confidence", 0.0))
            if confidence < self.confidence_threshold:
                continue

            x, y, w, h = result["box"]
            # MTCNN occasionally returns negative coordinates for faces
            # near the frame edge -- clamp instead of discarding outright.
            x, y = max(0, x), max(0, y)

            if not is_valid_bbox(x, y, w, h, frame_bgr.shape):
             continue
    
            if w < self.min_face_size or h < self.min_face_size:
             continue

            landmarks = result.get("keypoints")

            faces.append(
                DetectedFace(
                box=(x, y, w, h),
                confidence=confidence,
                landmarks=landmarks
            )
        )

    def detect_largest(self, frame_bgr: np.ndarray) -> Optional[DetectedFace]:
        """Convenience method for registration: return only the largest
        (closest) face in the frame, or None if no face is found."""
        faces = self.detect(frame_bgr)
        if not faces:
            return None
        return max(faces, key=lambda f: f.box[2] * f.box[3])

    @staticmethod
    def crop_face(frame_bgr: np.ndarray, face: DetectedFace, margin: float = 0.15) -> Optional[np.ndarray]:
        """Crop a face from the frame with a small margin around the
        MTCNN bounding box (improves VGGFace embedding quality)."""
        x, y, w, h = face.box
        mx, my = int(w * margin), int(h * margin)
        return safe_crop(frame_bgr, x - mx, y - my, w + 2 * mx, h + 2 * my)
