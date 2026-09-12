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

BOX_COLOR_KNOWN = (0, 200, 0)        # Green
BOX_COLOR_UNKNOWN = (0, 0, 220)      # Red
BOX_COLOR_ALREADY = (0, 165, 255)    # Orange


# ============================================================
# VIDEO CAMERA
# ============================================================

class VideoCamera:
    """
    Thread-safe webcam manager.

    - Camera opens only when start() is called
    - Camera closes physically when stop() is called
    - Background thread captures frames
    - Only latest frame is stored
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

            # Camera already running
            if self.is_running:
                return

            logger.info(
                "Opening camera %s...",
                self.camera_index
            )

            self._stop_event.clear()

            with self._frame_lock:
                self._frame = None
            # ------------------------------------------------
            # OPEN CAMERA
            # ------------------------------------------------

            try:

                cap = None

                if os.name == "nt":

                    # Try DirectShow first
                    cap = cv2.VideoCapture(
                        self.camera_index,
                        cv2.CAP_DSHOW
                    )

                    # If DirectShow fails, try MSMF
                    if not cap.isOpened():

                        cap.release()

                        cap = cv2.VideoCapture(
                            self.camera_index,
                            cv2.CAP_MSMF
                        )

                    # Final fallback
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

            except Exception:
                pass


            # ------------------------------------------------
            # REDUCE CAMERA BUFFER
            # ------------------------------------------------

            try:

                cap.set(
                    cv2.CAP_PROP_BUFFERSIZE,
                    1
                )

            except Exception:
                pass


            # ------------------------------------------------
            # CAMERA FPS
            # ------------------------------------------------

            try:

                cap.set(
                    cv2.CAP_PROP_FPS,
                    30
                )

            except Exception:
                pass


            # ------------------------------------------------
            # STORE CAMERA
            # ------------------------------------------------

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
        # WAIT BRIEFLY FOR FIRST FRAME
        # ----------------------------------------------------

        start_time = time.time()

        while time.time() - start_time < 1.5:

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


                # Read frame
                ok, frame = cap.read()


                # ------------------------------------------------
                # CAMERA READ FAILED
                # ------------------------------------------------

                if not ok or frame is None:

                    failures += 1

                    if failures >= 20:

                        logger.warning(
                            "Camera stopped returning frames."
                        )

                        break

                    time.sleep(0.02)

                    continue


                # ------------------------------------------------
                # CAMERA READ SUCCESS
                # ------------------------------------------------

                failures = 0


                # Keep ONLY newest frame
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

            # Already stopped
            if (
                not self._running
                and self.cap is None
            ):
                return


            logger.info(
                "Closing camera %s...",
                self.camera_index
            )


            # Stop thread
            self._running = False

            self._stop_event.set()


            # Save camera reference
            cap = self.cap

            self.cap = None


            # ------------------------------------------------
            # RELEASE PHYSICAL CAMERA
            # ------------------------------------------------

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


        # ----------------------------------------------------
        # JOIN THREAD OUTSIDE LOCK
        # ----------------------------------------------------

        thread = self._thread

        if (
            thread is not None
            and thread.is_alive()
            and thread != threading.current_thread()
        ):

            thread.join(timeout=1.0)


        self._thread = None


        # ----------------------------------------------------
        # CLEAR LAST FRAME
        # ----------------------------------------------------

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
    Face detection + face recognition + attendance pipeline.

    The AI models are loaded once and reused.
    """

    def __init__(self):

        logger.info(
            "Initializing recognition pipeline..."
        )


        # ----------------------------------------------------
        # LOAD COMPONENTS
        # ----------------------------------------------------

        self.detector = FaceDetector()

        self.recognizer = FaceRecognizer()

        self.attendance_manager = AttendanceManager()


        # ----------------------------------------------------
        # MODEL STATUS
        # ----------------------------------------------------

        self.model_ready = (
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
    # PROCESS FRAME
    # ========================================================

    def process_frame(
        self,
        frame: np.ndarray
    ) -> Tuple[
        np.ndarray,
        List[Dict[str, Any]]
    ]:

        # Copy frame for drawing
        annotated = frame.copy()

        results: List[
            Dict[str, Any]
        ] = []


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

            faces: List[DetectedFace] = (
                self.detector.detect(frame)
            )

        except Exception as exc:

            logger.warning(
                "Face detection error: %s",
                exc
            )

            return annotated, results


        # ----------------------------------------------------
        # PROCESS EACH FACE
        # ----------------------------------------------------

        for face in faces:

            x, y, w, h = face.box


            # Crop face
            crop = self.detector.crop_face(
                frame,
                face
            )


            if crop is None:
                continue


            # ------------------------------------------------
            # RECOGNIZE FACE
            # ------------------------------------------------

            try:

                (
                    recognition_status,
                    student_id,
                    confidence,
                ) = self.recognizer.identify(
                    crop
                )


            except ModelNotTrainedError:

                break


            except Exception as exc:

                logger.warning(
                    "Recognition error: %s",
                    exc
                )

                continue


            # ------------------------------------------------
            # KNOWN STUDENT
            # ------------------------------------------------

            if recognition_status == "known":

                try:

                    info = (
                        self.attendance_manager
                        .process_recognition(
                            student_id,
                            confidence
                        )
                    )

                except Exception as exc:

                    logger.warning(
                        "Attendance processing error: %s",
                        exc
                    )

                    continue


                # New attendance = green
                # Already marked = orange

                if info.get(
                    "newly_marked",
                    False
                ):

                    color = BOX_COLOR_KNOWN

                else:

                    color = BOX_COLOR_ALREADY


                name = info.get(
                    "name",
                    "Student"
                )


                attendance_status = info.get(
                    "status",
                    "Present"
                )


                label_lines = [

                    name,

                    f"ID: {student_id}",

                    attendance_status,

                    f"Confidence: "
                    f"{confidence * 100:.0f}%"

                ]


                results.append({

                    "student_id":
                        student_id,

                    "name":
                        name,

                    "status":
                        attendance_status,

                    "confidence":
                        round(confidence, 4),

                    "box":
                        [x, y, w, h],

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

                    "student_id":
                        None,

                    "name":
                        "Unknown",

                    "status":
                        "Unknown",

                    "confidence":
                        round(confidence, 4),

                    "box":
                        [x, y, w, h],

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

                line,

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
            frame_width - x
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

                line,

                (x + 6, text_y),

                font,

                font_scale,

                (255, 255, 255),

                thickness,

                cv2.LINE_AA

            )