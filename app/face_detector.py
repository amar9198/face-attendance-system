"""
app/face_detector.py
---------------------

MTCNN-based face detection.

Pipeline:

    Webcam frame
        ↓
    MTCNN Face Detector
        ↓
    List of DetectedFace objects

Used by:
    - Student face registration
    - Real-time face recognition
    - Attendance system
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app import config
from app.utils import get_logger, is_valid_bbox, safe_crop


# ============================================================
# LOGGER
# ============================================================

logger = get_logger(__name__)


# ============================================================
# DETECTED FACE DATA CLASS
# ============================================================

@dataclass
class DetectedFace:
    """
    Represents one detected face.
    """

    # Format:
    # (x, y, width, height)
    box: Tuple[int, int, int, int]

    confidence: float

    landmarks: Optional[
        Dict[str, Tuple[int, int]]
    ] = field(default=None)

    @property
    def crop_coords(self) -> Tuple[int, int, int, int]:
        """
        Return bounding box coordinates.
        """

        return self.box


# ============================================================
# FACE DETECTOR
# ============================================================

class FaceDetector:
    """
    Defensive wrapper around MTCNN.

    Features:
        - Lazy loading of MTCNN
        - Detect multiple faces
        - Confidence filtering
        - Minimum face size filtering
        - Safe bounding box validation
        - Detect largest face for registration
        - Safe face cropping
        - Always returns a list from detect()
    """

    def __init__(
        self,
        confidence_threshold: float = config.FACE_CONFIDENCE_THRESHOLD,
        min_face_size: int = config.MIN_FACE_SIZE,
    ):

        self.confidence_threshold = float(
            confidence_threshold
        )

        self.min_face_size = int(
            min_face_size
        )

        # MTCNN loads only when first needed
        self._detector = None
        self._fallback_detector = cv2.CascadeClassifier(
            os.path.join(
                cv2.data.haarcascades,
                "haarcascade_frontalface_default.xml",
            )
        )


    # ========================================================
    # LOAD MTCNN
    # ========================================================

    def _load(self):
        """
        Load the MTCNN detector lazily.
        """

        if self._detector is not None:
            return self._detector

        try:

            from mtcnn import MTCNN

        except ImportError as exc:

            logger.error(
                "MTCNN package is not installed."
            )

            raise ImportError(
                "The 'mtcnn' package is not installed. "
                "Install it using:\n"
                "pip install mtcnn tensorflow"
            ) from exc


        logger.info(
            "Loading MTCNN face detector..."
        )

        try:
            detector_options = {
                "steps_threshold": [0.5, 0.6, 0.7],
            }
            try:
                self._detector = MTCNN(
                    min_face_size=self.min_face_size,
                    **detector_options,
                )
            except TypeError as exc:
                if "min_face_size" not in str(exc):
                    raise
                logger.warning(
                    "Installed MTCNN does not support min_face_size; "
                    "using its default minimum face size."
                )
                self._detector = MTCNN(**detector_options)

        except Exception as exc:

            logger.exception(
                "Failed to initialize MTCNN."
            )

            raise RuntimeError(
                f"Could not initialize MTCNN: {exc}"
            ) from exc


        logger.info(
            "MTCNN loaded successfully."
        )

        return self._detector


    # ========================================================
    # DETECT ALL FACES
    # ========================================================

    def detect(
        self,
        frame_bgr: np.ndarray
    ) -> List[DetectedFace]:
        """
        Detect all faces in a BGR OpenCV frame.

        IMPORTANT:
        This function always returns a List[DetectedFace].

        Returns an empty list when:
            - Frame is invalid
            - No faces are detected
            - MTCNN fails
        """

        # ----------------------------------------------------
        # VALIDATE FRAME
        # ----------------------------------------------------

        if frame_bgr is None:

            logger.warning(
                "detect() received None frame."
            )

            return []


        if not isinstance(
            frame_bgr,
            np.ndarray
        ):

            logger.warning(
                "detect() received invalid frame type: %s",
                type(frame_bgr).__name__
            )

            return []


        if frame_bgr.size == 0:

            logger.warning(
                "detect() received empty frame."
            )

            return []


        # Need a normal image
        if frame_bgr.ndim != 3:

            logger.warning(
                "detect() received invalid frame dimensions."
            )

            return []


        # ----------------------------------------------------
        # LOAD DETECTOR
        # ----------------------------------------------------

        try:

            detector = self._load()

        except Exception as exc:

            logger.exception(
                "Could not load MTCNN: %s",
                exc
            )

            return self._detect_opencv_fallback(frame_bgr)


        # ----------------------------------------------------
        # CONVERT BGR -> RGB
        # ----------------------------------------------------

        try:

            frame_rgb = cv2.cvtColor(
                frame_bgr,
                cv2.COLOR_BGR2RGB
            )

        except Exception as exc:

            logger.warning(
                "Could not convert frame to RGB: %s",
                exc
            )

            return self._detect_opencv_fallback(frame_bgr)


        # ----------------------------------------------------
        # RUN MTCNN DETECTION
        # ----------------------------------------------------

        try:

            raw_results = detector.detect_faces(
                frame_rgb
            )

        except Exception as exc:

            logger.exception(
                "MTCNN detection failed: %s",
                exc
            )

            return self._detect_opencv_fallback(frame_bgr)


        # ----------------------------------------------------
        # VALIDATE RESULTS
        # ----------------------------------------------------

        if raw_results is None:

            return self._detect_opencv_fallback(frame_bgr)


        if not isinstance(
            raw_results,
            (list, tuple)
        ):

            logger.warning(
                "Unexpected MTCNN result type: %s",
                type(raw_results).__name__
            )

            return self._detect_opencv_fallback(frame_bgr)


        # ----------------------------------------------------
        # PROCESS DETECTIONS
        # ----------------------------------------------------

        faces: List[DetectedFace] = []

        frame_height, frame_width = (
            frame_bgr.shape[:2]
        )


        for result in raw_results:

            # ------------------------------------------------
            # VALIDATE RESULT
            # ------------------------------------------------

            if not isinstance(
                result,
                dict
            ):

                continue


            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            try:

                confidence = float(
                    result.get(
                        "confidence",
                        0.0
                    )
                )

            except (
                TypeError,
                ValueError
            ):

                continue


            # Ignore low-confidence detections
            if confidence < self.confidence_threshold:

                continue


            # ------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------

            box = result.get("box")


            if box is None:

                continue


            if not isinstance(
                box,
                (list, tuple, np.ndarray)
            ):

                continue


            if len(box) != 4:

                continue


            try:

                original_x = int(box[0])
                original_y = int(box[1])

                w = int(box[2])
                h = int(box[3])

            except (
                TypeError,
                ValueError,
                OverflowError
            ):

                continue


            # Width and height must be valid
            if w <= 0 or h <= 0:

                continue


            # ------------------------------------------------
            # CLAMP BOUNDING BOX TO FRAME
            # ------------------------------------------------

            # MTCNN may return negative coordinates.
            # Preserve the correct right/bottom edges.

            x1 = max(
                0,
                original_x
            )

            y1 = max(
                0,
                original_y
            )

            x2 = min(
                frame_width,
                original_x + w
            )

            y2 = min(
                frame_height,
                original_y + h
            )


            # Recalculate width and height after clamping

            x = int(x1)
            y = int(y1)

            w = int(x2 - x1)
            h = int(y2 - y1)


            if w <= 0 or h <= 0:

                continue


            # ------------------------------------------------
            # VALIDATE BOUNDING BOX
            # ------------------------------------------------

            try:

                valid = is_valid_bbox(
                    x,
                    y,
                    w,
                    h,
                    frame_bgr.shape
                )

            except Exception as exc:

                logger.warning(
                    "Bounding box validation failed: %s",
                    exc
                )

                valid = False


            if not valid:

                continue


            # ------------------------------------------------
            # MINIMUM FACE SIZE
            # ------------------------------------------------

            if w < self.min_face_size:

                continue


            if h < self.min_face_size:

                continue


            # ------------------------------------------------
            # LANDMARKS
            # ------------------------------------------------

            raw_landmarks = result.get(
                "keypoints"
            )

            landmarks = self._normalize_landmarks(
                raw_landmarks
            )


            # ------------------------------------------------
            # CREATE DETECTED FACE
            # ------------------------------------------------

            faces.append(

                DetectedFace(

                    box=(
                        int(x),
                        int(y),
                        int(w),
                        int(h)
                    ),

                    confidence=float(confidence),

                    landmarks=landmarks,

                )

            )


        # ====================================================
        # IMPORTANT: ALWAYS RETURN A LIST
        # ====================================================

        if faces:
            return faces

        return self._detect_opencv_fallback(frame_bgr)

    def _detect_opencv_fallback(
        self,
        frame_bgr: np.ndarray,
    ) -> List[DetectedFace]:
        """Use OpenCV's bundled detector when MTCNN misses a clear face."""
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            detections = self._fallback_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(self.min_face_size, self.min_face_size),
            )
        except Exception as exc:
            logger.warning("OpenCV fallback face detection failed: %s", exc)
            return []

        faces = []
        for x, y, width, height in detections:
            faces.append(
                DetectedFace(
                    box=(int(x), int(y), int(width), int(height)),
                    confidence=1.0,
                )
            )

        if faces:
            logger.info(
                "MTCNN found no usable face; OpenCV fallback detected %d face(s).",
                len(faces),
            )

        return faces


    # ========================================================
    # NORMALIZE LANDMARKS
    # ========================================================

    @staticmethod
    def _normalize_landmarks(
        landmarks: Any
    ) -> Optional[Dict[str, Tuple[int, int]]]:
        """
        Convert MTCNN landmarks into normal Python integers.
        """

        if not isinstance(
            landmarks,
            dict
        ):

            return None


        cleaned: Dict[
            str,
            Tuple[int, int]
        ] = {}


        for key, value in landmarks.items():

            try:

                if (
                    isinstance(
                        value,
                        (list, tuple, np.ndarray)
                    )
                    and len(value) >= 2
                ):

                    cleaned[str(key)] = (

                        int(value[0]),

                        int(value[1])

                    )

            except (
                TypeError,
                ValueError,
                OverflowError
            ):

                continue


        return cleaned or None


    # ========================================================
    # DETECT LARGEST FACE
    # ========================================================

    def detect_largest(
        self,
        frame_bgr: np.ndarray
    ) -> Optional[DetectedFace]:
        """
        Detect all faces and return the largest one.

        Used during student registration.

        Returns:
            DetectedFace if a face exists.

        Otherwise:
            None
        """

        faces = self.detect(
            frame_bgr
        )


        if not faces:

            return None


        return max(

            faces,

            key=lambda face: (
                face.box[2] * face.box[3]
            )

        )


    # ========================================================
    # CROP FACE
    # ========================================================

    @staticmethod
    def crop_face(
        frame_bgr: np.ndarray,
        face: DetectedFace,
        margin: float = 0.15,
    ) -> Optional[np.ndarray]:
        """
        Safely crop a detected face.

        A margin is added around the face because face
        recognition models generally perform better when
        some surrounding facial area is included.
        """

        # ----------------------------------------------------
        # VALIDATE FRAME
        # ----------------------------------------------------

        if frame_bgr is None:

            return None


        if not isinstance(
            frame_bgr,
            np.ndarray
        ):

            return None


        if frame_bgr.size == 0:

            return None


        # ----------------------------------------------------
        # VALIDATE FACE
        # ----------------------------------------------------

        if face is None:

            return None


        if not hasattr(
            face,
            "box"
        ):

            return None


        # ----------------------------------------------------
        # GET FACE BOX
        # ----------------------------------------------------

        try:

            x, y, w, h = face.box

            x = int(x)
            y = int(y)

            w = int(w)
            h = int(h)

        except (
            TypeError,
            ValueError,
            AttributeError
        ):

            return None


        if w <= 0 or h <= 0:

            return None


        # ----------------------------------------------------
        # VALIDATE MARGIN
        # ----------------------------------------------------

        try:

            margin = float(
                margin
            )

        except (
            TypeError,
            ValueError
        ):

            margin = 0.15


        # Prevent negative margin
        margin = max(
            0.0,
            margin
        )


        # ----------------------------------------------------
        # CALCULATE EXTRA SPACE
        # ----------------------------------------------------

        margin_x = int(
            w * margin
        )

        margin_y = int(
            h * margin
        )


        crop_x = x - margin_x

        crop_y = y - margin_y

        crop_w = w + (
            2 * margin_x
        )

        crop_h = h + (
            2 * margin_y
        )


        # ----------------------------------------------------
        # SAFE CROP
        # ----------------------------------------------------

        try:

            crop = safe_crop(

                frame_bgr,

                crop_x,

                crop_y,

                crop_w,

                crop_h,

            )


            if crop is None:

                return None


            if not isinstance(
                crop,
                np.ndarray
            ):

                return None


            if crop.size == 0:

                return None


            return crop


        except Exception as exc:

            logger.warning(
                "Face cropping failed: %s",
                exc
            )

            return None