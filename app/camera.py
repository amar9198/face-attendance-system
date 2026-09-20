"""
app/camera.py
-------------

Camera manager and face recognition pipeline.

Features:
- Camera opens only when start() is called
- Camera physically releases when stop() is called
- Background thread captures latest frame
- Low latency camera streaming
- Multiple face recognition
- Attendance processing
- Thread-safe camera handling
"""

import os
import threading
import time
from typing import Optional, List, Dict, Any, Tuple

import cv2
import numpy as np

from app import config
from app.attendance import AttendanceManager
from app.face_detector import FaceDetector, DetectedFace
from app.face_recognizer import FaceRecognizer

from app.utils import (
    get_logger,
    CameraUnavailableError,
    ModelNotTrainedError,
)


# ============================================================
# LOGGER
# ============================================================

logger = get_logger(__name__)


# ============================================================
# COLORS (BGR)
# ============================================================

BOX_COLOR_KNOWN = (0, 200, 0)
BOX_COLOR_UNKNOWN = (0, 0, 220)
BOX_COLOR_ALREADY = (0, 165, 255)


# ============================================================
# VIDEO CAMERA
# ============================================================

class VideoCamera:
    """
    Thread-safe webcam manager.
    """

    def __init__(
        self,
        camera_index: int = config.CAMERA_INDEX,
    ):

        self.camera_index = camera_index

        self.cap: Optional[cv2.VideoCapture] = None

        self._frame: Optional[np.ndarray] = None

        self._frame_lock = threading.Lock()

        self._camera_lock = threading.Lock()

        self._running = False

        self._thread: Optional[threading.Thread] = None

        self._stop_event = threading.Event()


    # ========================================================
    # START CAMERA
    # ========================================================

    def start(self) -> None:

        with self._camera_lock:

            if self.is_running:
                return

            logger.info(
                "Opening camera %s...",
                self.camera_index
            )

            self._stop_event.clear()

            with self._frame_lock:
                self._frame = None

            cap = None

            try:

                if os.name == "nt":

                    cap = cv2.VideoCapture(
                        self.camera_index,
                        cv2.CAP_DSHOW
                    )

                    if not cap.isOpened():

                        cap.release()

                        cap = cv2.VideoCapture(
                            self.camera_index,
                            cv2.CAP_MSMF
                        )

                    if not cap.isOpened():

                        cap.release()

                        cap = cv2.VideoCapture(
                            self.camera_index
                        )

                else:

                    cap = cv2.VideoCapture(
                        self.camera_index
                    )

            except Exception as exc:

                raise CameraUnavailableError(
                    f"Could not initialize camera: {exc}"
                )


            # ------------------------------------------------
            # CHECK CAMERA
            # ------------------------------------------------

            if cap is None or not cap.isOpened():

                if cap is not None:

                    try:
                        cap.release()
                    except Exception:
                        pass

                raise CameraUnavailableError(
                    f"Could not open camera at index "
                    f"{self.camera_index}."
                )


            # ------------------------------------------------
            # CAMERA SETTINGS
            # ------------------------------------------------

            try:

                cap.set(
                    cv2.CAP_PROP_FRAME_WIDTH,
                    config.CAMERA_FRAME_WIDTH
                )

                cap.set(
                    cv2.CAP_PROP_FRAME_HEIGHT,
                    config.CAMERA_FRAME_HEIGHT
                )

                cap.set(
                    cv2.CAP_PROP_BUFFERSIZE,
                    1
                )

                cap.set(
                    cv2.CAP_PROP_FPS,
                    30
                )

            except Exception:
                pass


            self.cap = cap

            self._running = True


            # ------------------------------------------------
            # START BACKGROUND THREAD
            # ------------------------------------------------

            self._thread = threading.Thread(
                target=self._update_loop,
                daemon=True,
                name="CameraThread"
            )

            self._thread.start()


        # ----------------------------------------------------
        # WAIT FOR FIRST FRAME
        # ----------------------------------------------------

        start_time = time.time()

        while time.time() - start_time < 2:

            with self._frame_lock:

                if self._frame is not None:

                    logger.info(
                        "Camera %s ready.",
                        self.camera_index
                    )

                    return

            if not self._running:
                break

            time.sleep(0.01)


        logger.warning(
            "Camera opened but first frame is not ready yet."
        )


    # ========================================================
    # BACKGROUND CAMERA LOOP
    # ========================================================

    def _update_loop(self) -> None:

        failures = 0

        try:

            while (
                self._running
                and not self._stop_event.is_set()
            ):

                cap = self.cap

                if cap is None:
                    break


                ok, frame = cap.read()


                if not ok or frame is None:

                    failures += 1

                    if failures >= 20:

                        logger.warning(
                            "Camera stopped returning frames."
                        )

                        break

                    time.sleep(0.02)

                    continue


                failures = 0


                with self._frame_lock:

                    self._frame = frame


        except Exception as exc:

            logger.warning(
                "Camera thread error: %s",
                exc
            )


        finally:

            self._running = False


    # ========================================================
    # READ LATEST FRAME
    # ========================================================

    def read(self) -> Optional[np.ndarray]:

        with self._frame_lock:

            if self._frame is None:
                return None

            return self._frame.copy()


    # ========================================================
    # STOP CAMERA
    # ========================================================

    def stop(self) -> None:

        with self._camera_lock:

            if (
                not self._running
                and self.cap is None
            ):
                return


            logger.info(
                "Closing camera %s...",
                self.camera_index
            )


            self._running = False

            self._stop_event.set()


            cap = self.cap

            self.cap = None


            if cap is not None:

                try:

                    cap.release()

                    logger.info(
                        "Physical camera released."
                    )

                except Exception as exc:

                    logger.warning(
                        "Camera release error: %s",
                        exc
                    )


        thread = self._thread

        if (
            thread is not None
            and thread.is_alive()
            and thread != threading.current_thread()
        ):

            thread.join(timeout=1.0)


        self._thread = None


        with self._frame_lock:

            self._frame = None


        logger.info(
            "Camera closed successfully."
        )


    # ========================================================
    # CAMERA STATUS
    # ========================================================

    @property
    def is_running(self) -> bool:

        cap = self.cap

        return (

            self._running

            and cap is not None

            and cap.isOpened()

        )


# ============================================================
# RECOGNITION PIPELINE
# ============================================================

class RecognitionPipeline:
    """
    Face detection + recognition + attendance pipeline.
    """

    def __init__(self):

        logger.info(
            "Initializing recognition pipeline..."
        )


        self.detector = FaceDetector()

        self.recognizer = FaceRecognizer()

        self.attendance_manager = AttendanceManager()


        self.model_ready = bool(
            self.recognizer.is_trained
        )


        if self.model_ready:

            logger.info(
                "Recognition pipeline ready."
            )

        else:

            logger.warning(
                "Recognition model is not trained."
            )


    # ========================================================
    # SAFE FACE LIST
    # ========================================================

    @staticmethod
    def _normalize_faces(faces):

        """
        Always return a Python list.

        Prevents:
        TypeError: 'NoneType' object is not iterable
        """

        if faces is None:
            return []

        if isinstance(faces, list):
            return faces

        if isinstance(faces, tuple):
            return list(faces)

        try:
            return list(faces)

        except TypeError:
            return [faces]


    # ========================================================
    # SAFE RESULT DICTIONARY
    # ========================================================

    @staticmethod
    def _safe_dict(value):

        if isinstance(value, dict):
            return value

        return {}


    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(
        self,
        frame: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        return process_frame(self, frame)


def process_frame(
    self,
    frame: np.ndarray
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:

    # ----------------------------------------------------
    # VALIDATE FRAME
    # ----------------------------------------------------

    if frame is None:

        logger.warning("Received empty frame.")

        return np.zeros(
            (480, 640, 3),
            dtype=np.uint8
        ), []


    # Copy frame for drawing

    annotated = frame.copy()

    results: List[Dict[str, Any]] = []


    # ----------------------------------------------------
    # CHECK TRAINED MODEL
    # ----------------------------------------------------

    if not self.recognizer.is_trained:

        cv2.putText(

            annotated,

            "Model not trained",

            (10, 30),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 0, 255),

            2,

            cv2.LINE_AA

        )

        return annotated, results


    # ----------------------------------------------------
    # DETECT ALL FACES
    # ----------------------------------------------------

    try:

        faces = self.detector.detect(frame)


        # IMPORTANT FIX:
        # Some detectors return None when no face is found

        if faces is None:

            faces = []


        # Convert safely to list

        if not isinstance(faces, (list, tuple)):

            try:

                faces = list(faces)

            except TypeError:

                logger.warning(
                    "Face detector returned invalid value: %s",
                    type(faces).__name__
                )

                faces = []


    except Exception as exc:

        logger.exception(
            "Face detection error: %s",
            exc
        )

        return annotated, results


    # ----------------------------------------------------
    # NO FACE DETECTED
    # ----------------------------------------------------

    if len(faces) == 0:

        return annotated, results


    # ----------------------------------------------------
    # PROCESS EACH FACE
    # ----------------------------------------------------

    for face in faces:

        try:

            # ------------------------------------------------
            # VALIDATE FACE
            # ------------------------------------------------

            if face is None:

                continue


            if not hasattr(face, "box"):

                logger.warning(
                    "Invalid detected face object."
                )

                continue


            x, y, w, h = face.box


            # Convert coordinates safely

            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)


            if w <= 0 or h <= 0:

                continue


            # ------------------------------------------------
            # CROP FACE
            # ------------------------------------------------

            crop = self.detector.crop_face(
                frame,
                face
            )


            if crop is None:

                continue


            if crop.size == 0:

                continue


            # ------------------------------------------------
            # RECOGNIZE FACE
            # ------------------------------------------------

            try:

                recognition_status, student_id, confidence = (
                    self.recognizer.identify(crop)
                )


            except ModelNotTrainedError:

                logger.warning(
                    "Recognition model is not trained."
                )

                return annotated, results


            except Exception as exc:

                logger.warning(
                    "Recognition error: %s",
                    exc
                )

                continue


            # ------------------------------------------------
            # SAFE CONFIDENCE
            # ------------------------------------------------

            try:

                confidence = float(confidence)

            except (TypeError, ValueError):

                confidence = 0.0


            # ------------------------------------------------
            # KNOWN STUDENT
            # ------------------------------------------------

            if recognition_status == "known":

                name = self.attendance_manager.get_student_name(student_id)
                attendance_allowed = (
                    confidence >= config.ATTENDANCE_THRESHOLD
                )

                if attendance_allowed:
                    try:
                        info = (
                            self.attendance_manager
                            .process_recognition(
                                student_id,
                                confidence
                            )
                        )


                        # Ensure info is always a dictionary

                        if info is None:
                            info = {}


                    except Exception as exc:

                        logger.exception(
                            "Attendance processing error: %s",
                            exc
                        )

                        info = {}
                else:
                    info = {
                        "name": name,
                        "status": "Match - attendance not marked",
                        "newly_marked": False,
                    }


                # ------------------------------------------------
                # COLOR
                # ------------------------------------------------

                if info.get(
                    "newly_marked",
                    False
                ):

                    color = BOX_COLOR_KNOWN

                else:

                    color = BOX_COLOR_ALREADY


                # ------------------------------------------------
                # STUDENT INFORMATION
                # ------------------------------------------------

                name = info.get(
                    "name",
                    name
                )


                attendance_status = info.get(
                    "status",
                    "Present"
                )


                label_lines = [

                    str(name),

                    f"ID: {student_id}",

                    str(attendance_status),

                    f"Confidence: "
                    f"{confidence * 100:.0f}%"

                ]


                results.append({

                    "student_id": str(student_id),

                    "name": str(name),

                    "status": str(attendance_status),

                    "confidence": round(
                        confidence,
                        4
                    ),

                    "box": [

                        x,
                        y,
                        w,
                        h

                    ]

                })


            # ------------------------------------------------
            # UNKNOWN FACE
            # ------------------------------------------------

            else:

                color = BOX_COLOR_UNKNOWN


                label_lines = [

                    "Unknown",

                    f"Confidence: "
                    f"{confidence * 100:.0f}%"

                ]


                results.append({

                    "student_id": None,

                    "name": "Unknown",

                    "status": "Unknown",

                    "confidence": round(
                        confidence,
                        4
                    ),

                    "box": [

                        x,
                        y,
                        w,
                        h

                    ]

                })


            # ------------------------------------------------
            # DRAW RESULT
            # ------------------------------------------------

            self._draw_annotation(

                annotated,

                x,
                y,
                w,
                h,

                color,

                label_lines

            )


        except Exception as exc:

            # IMPORTANT:
            # One bad face should not crash recognition

            logger.exception(
                "Error processing detected face: %s",
                exc
            )

            continue


    return annotated, results


    # ========================================================
    # DRAW FACE ANNOTATION
    # ========================================================

    @staticmethod
    def _draw_annotation(

        frame: np.ndarray,

        x: int,
        y: int,
        w: int,
        h: int,

        color: Tuple[int, int, int],

        label_lines: List[str],

    ) -> None:


        if frame is None:
            return


        frame_height, frame_width = (
            frame.shape[:2]
        )


        # ----------------------------------------------------
        # KEEP FACE BOX INSIDE FRAME
        # ----------------------------------------------------

        x = max(
            0,
            min(x, frame_width - 1)
        )

        y = max(
            0,
            min(y, frame_height - 1)
        )

        w = max(
            1,
            min(w, frame_width - x)
        )

        h = max(
            1,
            min(h, frame_height - y)
        )


        # ----------------------------------------------------
        # DRAW FACE RECTANGLE
        # ----------------------------------------------------

        cv2.rectangle(

            frame,

            (x, y),

            (x + w, y + h),

            color,

            2

        )


        # ----------------------------------------------------
        # TEXT SETTINGS
        # ----------------------------------------------------

        font = cv2.FONT_HERSHEY_SIMPLEX

        font_scale = 0.5

        thickness = 1

        line_height = 18


        # ----------------------------------------------------
        # CALCULATE LABEL WIDTH
        # ----------------------------------------------------

        label_width = w


        for line in label_lines:

            text_size, _ = cv2.getTextSize(

                str(line),

                font,

                font_scale,

                thickness

            )


            label_width = max(

                label_width,

                text_size[0] + 12

            )


        label_height = (

            line_height
            * len(label_lines)
            + 8

        )


        # ----------------------------------------------------
        # LABEL POSITION
        # ----------------------------------------------------

        text_top = (
            y - label_height
        )


        # Put below face if not enough space above

        if text_top < 0:

            text_top = (
                y + h + 2
            )


        # Keep inside bottom

        if text_top + label_height > frame_height:

            text_top = max(

                0,

                frame_height - label_height

            )


        # Keep label width inside frame

        label_width = min(

            label_width,

            max(1, frame_width - x)

        )


        # ----------------------------------------------------
        # DRAW LABEL BACKGROUND
        # ----------------------------------------------------

        cv2.rectangle(

            frame,

            (x, text_top),

            (

                x + label_width,

                text_top + label_height

            ),

            color,

            cv2.FILLED

        )


        # ----------------------------------------------------
        # DRAW TEXT
        # ----------------------------------------------------

        for i, line in enumerate(label_lines):

            text_y = (

                text_top

                + 16

                + i * line_height

            )


            cv2.putText(

                frame,

                str(line),

                (x + 6, text_y),

                font,

                font_scale,

                (255, 255, 255),

                thickness,

                cv2.LINE_AA

            )