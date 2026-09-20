"""
app/main.py
-----------

Flask application for the Face Attendance System.
"""

import atexit
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from app import config, database
from app.camera import VideoCamera, RecognitionPipeline
from app.face_detector import FaceDetector

from app.utils import (
    get_logger,
    sanitize_student_id,
    sanitize_name,
    CameraUnavailableError,
    ModelNotTrainedError,
)


# ============================================================
# LOGGER
# ============================================================

logger = get_logger(__name__)


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "templates"),
    static_folder=os.path.join(PROJECT_ROOT, "static"),
)

app.secret_key = config.SECRET_KEY


# ============================================================
# GLOBAL STATE
# ============================================================

_camera = None
_pipeline = None

_registration_detector = FaceDetector()

_latest_results = []

_results_lock = threading.Lock()
_live_processing_lock = threading.Lock()
_browser_liveness_lock = threading.Lock()
_browser_liveness = {
    "eyes_were_open": False,
    "blink_count": 0,
    "valid_until": 0.0,
}

_eye_cascade = cv2.CascadeClassifier(
    os.path.join(
        cv2.data.haarcascades,
        "haarcascade_eye.xml",
    )
)
_face_cascade = cv2.CascadeClassifier(
    os.path.join(
        cv2.data.haarcascades,
        "haarcascade_frontalface_default.xml",
    )
)


def detect_browser_blink(frame):
    """Return whether the face currently appears to have closed eyes."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(40, 40),
    )
    if len(faces) == 0:
        return None

    x, y, width, height = max(
        faces,
        key=lambda detected: detected[2] * detected[3],
    )
    upper_face = gray[y:y + max(1, int(height * 0.6)), x:x + width]
    eyes = _eye_cascade.detectMultiScale(
        upper_face,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(6, 6),
    )
    return len(eyes) == 0


# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

FRAME_SKIP = 2
MAX_PROCESS_WIDTH = 960
JPEG_QUALITY = 80


# ============================================================
# JSON SAFE CONVERSION
# ============================================================

def make_json_safe(value):
    """
    Convert NumPy values and other non-JSON values into
    normal Python objects.
    """

    if value is None:
        return None

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, list):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            make_json_safe(item)
            for item in value
        ]

    return value


# ============================================================
# RESULT NORMALIZATION
# ============================================================

def normalize_results(results):
    """
    Always return recognition results as a Python list.

    Handles:
    - None
    - list
    - tuple
    - dict
    - NumPy arrays
    - single recognition objects

    This prevents:
    TypeError: 'NoneType' object is not iterable
    """

    if results is None:
        return []

    # NumPy array
    if isinstance(results, np.ndarray):
        results = results.tolist()

    # Dictionary
    if isinstance(results, dict):

        # Common API structure:
        # {"results": [...]}
        if "results" in results:
            return normalize_results(results.get("results"))

        # Single recognition result
        return [make_json_safe(results)]

    # List
    if isinstance(results, list):

        cleaned = []

        for item in results:

            if item is None:
                continue

            if isinstance(item, dict):
                cleaned.append(make_json_safe(item))

            elif isinstance(item, np.ndarray):
                cleaned.extend(
                    normalize_results(item)
                )

            else:
                cleaned.append(
                    make_json_safe(item)
                )

        return cleaned

    # Tuple
    if isinstance(results, tuple):

        # If tuple contains recognition objects
        if len(results) == 0:
            return []

        # Do NOT blindly assume tuple is iterable recognition data
        return [
            make_json_safe(item)
            for item in results
            if item is not None
        ]

    # String should be a single result, not split into characters
    if isinstance(results, str):
        return [results]

    # Any other object
    return [make_json_safe(results)]


# ============================================================
# EXTRACT RESULTS FROM PIPELINE
# ============================================================

def extract_pipeline_results(processed):
    """
    Safely extract recognition results from the output of:

        pipeline.process_frame(frame)

    Expected common format:

        (annotated_frame, results)

    But safely handles other formats too.
    """

    if processed is None:
        return []

    # Normal format:
    # annotated_frame, results
    if isinstance(processed, tuple):

        if len(processed) >= 2:
            return normalize_results(processed[1])

        return []

    # Dictionary
    if isinstance(processed, dict):
        return normalize_results(processed)

    # List
    if isinstance(processed, list):
        return normalize_results(processed)

    # NumPy array is probably an image, not recognition results
    if isinstance(processed, np.ndarray):
        return []

    return normalize_results(processed)


# ============================================================
# CAMERA FUNCTIONS
# ============================================================

def get_camera():

    global _camera

    if _camera is not None:

        if getattr(_camera, "is_running", False):
            return _camera

    logger.info("Creating camera instance...")

    _camera = VideoCamera()

    try:

        _camera.start()

    except Exception:

        _camera = None
        raise

    return _camera


def get_pipeline():

    global _pipeline

    if _pipeline is None:

        logger.info("Loading recognition pipeline...")

        _pipeline = RecognitionPipeline()

    return _pipeline


def release_camera():

    global _camera

    if _camera is not None:

        try:

            logger.info("Releasing camera...")

            _camera.stop()

        except Exception as exc:

            logger.warning(
                "Camera release error: %s",
                exc
            )

        finally:

            _camera = None


atexit.register(release_camera)


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

database.init_db()


# ============================================================
# RESIZE FRAME
# ============================================================

def resize_for_processing(frame):

    if frame is None:
        return None, 1.0

    height, width = frame.shape[:2]

    if width <= MAX_PROCESS_WIDTH:
        return frame, 1.0

    scale = MAX_PROCESS_WIDTH / float(width)

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    return resized, scale


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return render_template(
        "error.html",
        message="Page not found."
    ), 404


@app.errorhandler(500)
def server_error(error):

    logger.exception(
        "Server error: %s",
        error
    )

    return render_template(
        "error.html",
        message="An unexpected error occurred."
    ), 500


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    stats = database.get_dashboard_stats()

    try:

        pipeline = get_pipeline()

        model_ready = getattr(
            getattr(pipeline, "recognizer", None),
            "is_trained",
            False
        )

    except Exception:

        model_ready = False

    return render_template(
        "index.html",
        stats=stats,
        model_ready=model_ready,
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    stats = database.get_dashboard_stats()

    recent = database.get_attendance()[:10]

    camera_running = (
        _camera is not None
        and getattr(_camera, "is_running", False)
    )

    try:

        pipeline = get_pipeline()

        model_ready = getattr(
            getattr(pipeline, "recognizer", None),
            "is_trained",
            False
        )

    except Exception:

        model_ready = False

    return render_template(
        "dashboard.html",
        stats=stats,
        recent=recent,
        camera_running=camera_running,
        model_ready=model_ready,
    )


# ============================================================
# STUDENTS
# ============================================================

@app.route("/students")
def students():

    all_students = database.get_all_students()

    return render_template(
        "students.html",
        students=all_students,
    )


# ============================================================
# DELETE STUDENT
# ============================================================

@app.route(
    "/students/delete/<student_id>",
    methods=["POST"]
)
def delete_student(student_id):

    try:

        student = database.get_student(student_id)

        if student is None:

            flash(
                "Student not found.",
                "danger"
            )

            return redirect(
                url_for("students")
            )

        safe_student_id = sanitize_student_id(student_id)
        deleted_folders = []

        if os.path.isdir(config.DATASET_PATH):
            for folder_name in os.listdir(config.DATASET_PATH):
                folder_path = os.path.join(config.DATASET_PATH, folder_name)
                if (
                    os.path.isdir(folder_path)
                    and (
                        folder_name == safe_student_id
                        or folder_name.startswith(f"{safe_student_id}_")
                    )
                ):
                    shutil.rmtree(folder_path)
                    deleted_folders.append(folder_name)

        database.delete_student(student_id)

        cache_removed_entries = 0
        if os.path.isfile(config.EMBEDDING_CACHE_PATH):
            try:
                with open(
                    config.EMBEDDING_CACHE_PATH,
                    "r",
                    encoding="utf-8",
                ) as cache_file:
                    cache_data = json.load(cache_file)

                cached_images = cache_data.get("images", {})
                retained_images = {}
                for image_path, entry in cached_images.items():
                    cached_student_id = str(entry.get("student_id", ""))
                    if cached_student_id == safe_student_id:
                        cache_removed_entries += 1
                    else:
                        retained_images[image_path] = entry

                cache_data["images"] = retained_images
                with open(
                    config.EMBEDDING_CACHE_PATH,
                    "w",
                    encoding="utf-8",
                ) as cache_file:
                    json.dump(cache_data, cache_file)
            except (OSError, ValueError, TypeError) as exc:
                logger.warning(
                    "Could not prune embedding cache during deletion: %s",
                    exc,
                )

        model_artifacts = (
            config.SVM_MODEL_PATH,
            config.LABEL_ENCODER_PATH,
            config.EMBEDDINGS_PATH,
            config.LABELS_PATH,
            config.METADATA_PATH,
            os.path.join(config.MODEL_DIR, "test_split.npz"),
        )
        deleted_model_artifacts = []
        for artifact_path in model_artifacts:
            if os.path.isfile(artifact_path):
                os.remove(artifact_path)
                deleted_model_artifacts.append(artifact_path)

        global _pipeline
        _pipeline = None

        logger.info(
            "Student deleted: %s; folders=%s; cache_entries_removed=%s; "
            "model_artifacts_removed=%s",
            student_id,
            deleted_folders,
            cache_removed_entries,
            len(deleted_model_artifacts),
        )

        flash(
            f"Student {student_id} and all captured face data were deleted. "
            "Train the model again to recognize the remaining students.",
            "success"
        )

    except Exception as exc:

        logger.exception(
            "Student deletion failed"
        )

        flash(
            f"Could not delete student: {str(exc)}",
            "danger"
        )

    return redirect(
        url_for("students")
    )


# ============================================================
# STUDENT REGISTRATION
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "POST":

        try:

            student_id = sanitize_student_id(
                request.form.get(
                    "student_id",
                    ""
                )
            )

            name = sanitize_name(
                request.form.get(
                    "name",
                    ""
                )
            )

            department = request.form.get(
                "department",
                ""
            ).strip()

            email = request.form.get(
                "email",
                ""
            ).strip()

            if not student_id:

                flash(
                    "Student ID is required.",
                    "danger"
                )

                return redirect(
                    url_for("register")
                )

            if not name:

                flash(
                    "Student name is required.",
                    "danger"
                )

                return redirect(
                    url_for("register")
                )

            existing_student = database.get_student(
                student_id
            )

            if existing_student is not None:

                flash(
                    f"Student {student_id} already exists.",
                    "warning"
                )

                return redirect(
                    url_for("register")
                )

            success = database.add_student(
                student_id,
                name,
                department,
                email
            )

            if not success:

                flash(
                    "Student could not be registered.",
                    "danger"
                )

                return redirect(
                    url_for("register")
                )

            logger.info(
                "Student registered successfully: %s",
                student_id
            )

            flash(
                "Student registered successfully. "
                "Now capture face images.",
                "success"
            )

            return redirect(
                url_for(
                    "capture",
                    student_id=student_id
                )
            )

        except Exception as exc:

            logger.exception(
                "Student registration failed"
            )

            flash(
                f"Registration failed: {str(exc)}",
                "danger"
            )

            return redirect(
                url_for("register")
            )

    return render_template(
        "register.html"
    )


# ============================================================
# FACE CAPTURE PAGE
# ============================================================

@app.route("/capture/<student_id>")
def capture(student_id):

    student = database.get_student(student_id)

    if student is None:

        flash(
            "Student not found.",
            "danger"
        )

        return redirect(
            url_for("students")
        )

    requested_target = request.args.get("target_count", config.NUM_COLLECTION_IMAGES, type=int)
    target_count = max(
        config.MIN_COLLECTION_IMAGES,
        min(requested_target, config.MAX_COLLECTION_IMAGES),
    )

    safe_name = sanitize_name(student.get("name", "Unknown"))
    folder_name = f"{student_id}_{safe_name}"
    folder_path = os.path.join(config.DATASET_PATH, folder_name)
    image_extensions = {".jpg", ".jpeg", ".png"}
    existing_count = 0

    if os.path.isdir(folder_path):
        existing_count = sum(
            1
            for filename in os.listdir(folder_path)
            if os.path.splitext(filename)[1].lower() in image_extensions
        )

    return render_template(
        "register.html",
        capture_mode=True,
        name=student.get("name", ""),
        student_id=student_id,
        folder_name=folder_name,
        existing_count=existing_count,
        target_count=target_count,
    )


# ============================================================
# API FACE CAPTURE
# ============================================================

@app.route(
    "/api/capture/<student_id>",
    methods=["POST"]
)
def api_capture(student_id):

    try:

        student = database.get_student(student_id)

        if student is None:

            return jsonify({

                "success": False,

                "message":
                    f"Student {student_id} not found."

            }), 404

        requested_target = request.form.get("target_count", type=int)
        if requested_target is None:
            requested_target = config.NUM_COLLECTION_IMAGES

        target_count = max(
            config.MIN_COLLECTION_IMAGES,
            min(requested_target, config.MAX_COLLECTION_IMAGES),
        )

        folder_name = request.form.get(
            "folder_name"
        )

        if not folder_name:

            student_name = student.get(
                "name",
                "Unknown"
            )

            safe_name = (
                student_name
                .replace(" ", "_")
                .replace("/", "_")
                .replace("\\", "_")
            )

            folder_name = (
                f"{student_id}_{safe_name}"
            )

        folder_path = os.path.join(
            config.DATASET_PATH,
            folder_name
        )

        os.makedirs(
            folder_path,
            exist_ok=True
        )

        if "image" not in request.files:

            return jsonify({

                "success": False,

                "message":
                    "No image received from camera."

            }), 400

        image_file = request.files["image"]

        image_bytes = image_file.read()

        if not image_bytes:

            return jsonify({

                "success": False,

                "message":
                    "Empty image received."

            }), 400

        np_array = np.frombuffer(
            image_bytes,
            np.uint8
        )

        frame = cv2.imdecode(
            np_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            return jsonify({

                "success": False,

                "message":
                    "Invalid camera image."

            }), 400

        face = (
            _registration_detector
            .detect_largest(frame)
        )

        if face is None:

            return jsonify({

                "success": False,

                "message":
                    "No face detected. "
                    "Please look directly at the camera."

            })

        crop = (
            _registration_detector
            .crop_face(
                frame,
                face
            )
        )

        if crop is None:

            return jsonify({

                "success": False,

                "message":
                    "Could not crop detected face."

            })

        crop_resized = cv2.resize(
            crop,
            config.IMAGE_SIZE,
            interpolation=cv2.INTER_AREA
        )

        existing = [

            f for f in os.listdir(folder_path)

            if f.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )
            )

        ]

        next_index = len(existing) + 1

        filename = f"{next_index:03d}.jpg"

        output_path = os.path.join(
            folder_path,
            filename
        )

        success = cv2.imwrite(
            output_path,
            crop_resized
        )

        if not success:

            return jsonify({

                "success": False,

                "message":
                    "Could not save captured image."

            }), 500

        done = next_index >= target_count

        logger.info(
            "Image captured: %s (%s/%s)",
            student_id,
            next_index,
            target_count
        )

        return jsonify({

            "success": True,

            "count": next_index,

            "target":
                target_count,

            "done": done,

            "message":
                f"Captured {next_index}/"
                f"{target_count}"

        })

    except Exception as exc:

        logger.exception(
            "Browser face capture error"
        )

        return jsonify({

            "success": False,

            "message": str(exc)

        }), 500


# ============================================================
# MODEL TRAINING
# ============================================================

@app.route(
    "/api/train",
    methods=["POST"]
)
def api_train():

    global _pipeline

    training_started = time.perf_counter()
    release_camera()

    project_python = os.path.join(
        PROJECT_ROOT,
        "venv",
        "Scripts",
        "python.exe",
    )
    python_exe = (
        project_python
        if os.path.isfile(project_python)
        else sys.executable
    )

    try:

        extract = subprocess.run(

            [
                python_exe,

                os.path.join(
                    PROJECT_ROOT,
                    "scripts",
                    "extract_features.py"
                )
            ],

            cwd=PROJECT_ROOT,

            capture_output=True,

            text=True,

            timeout=3600,

        )

        if extract.returncode != 0:

            logger.error(
                "Feature extraction failed: %s",
                extract.stderr
            )

            return jsonify({

                "success": False,

                "stage":
                    "extract_features",

                "log":
                    extract.stderr[-4000:],

                "duration_seconds":
                    round(time.perf_counter() - training_started, 2)

            }), 500

        train = subprocess.run(

            [
                python_exe,

                os.path.join(
                    PROJECT_ROOT,
                    "scripts",
                    "train_model.py"
                )
            ],

            cwd=PROJECT_ROOT,

            capture_output=True,

            text=True,

            timeout=3600,

        )

        if train.returncode != 0:

            logger.error(
                "Training failed: %s",
                train.stderr
            )

            return jsonify({

                "success": False,

                "stage":
                    "train_model",

                "log":
                    train.stderr[-4000:],

                "duration_seconds":
                    round(time.perf_counter() - training_started, 2)

            }), 500

    except subprocess.TimeoutExpired:

        return jsonify({

            "success": False,

            "message":
                "Training timed out.",

            "duration_seconds":
                round(time.perf_counter() - training_started, 2)

        }), 500

    except Exception as exc:

        logger.exception(
            "Training failed"
        )

        return jsonify({

            "success": False,

            "message": str(exc),

            "duration_seconds":
                round(time.perf_counter() - training_started, 2)

        }), 500

    # Force pipeline reload
    _pipeline = None

    return jsonify({

        "success": True,

        "log":
            extract.stdout[-2000:]
            + "\n"
            + train.stdout[-2000:],

        "duration_seconds":
            round(time.perf_counter() - training_started, 2)

    })


# ============================================================
# LIVE ATTENDANCE PAGE
# ============================================================

@app.route("/attendance/live")
def live_attendance():

    try:

        pipeline = get_pipeline()

        model_ready = getattr(
            getattr(pipeline, "recognizer", None),
            "is_trained",
            False
        )

    except Exception:

        model_ready = False

    camera_running = (

        _camera is not None

        and getattr(
            _camera,
            "is_running",
            False
        )

    )

    return render_template(

        "attendance.html",

        model_ready=model_ready,

        camera_running=camera_running,

        history_mode=False,

    )


# ============================================================
# START CAMERA
# ============================================================

@app.route(
    "/api/camera/start",
    methods=["POST"]
)
def api_camera_start():

    try:

        cam = get_camera()

        if not getattr(
            cam,
            "is_running",
            False
        ):

            cam.start()

        return jsonify({

            "success": True

        })

    except CameraUnavailableError as exc:

        return jsonify({

            "success": False,

            "message": str(exc)

        }), 503

    except Exception as exc:

        logger.exception(
            "Camera start error"
        )

        return jsonify({

            "success": False,

            "message": str(exc)

        }), 500


# ============================================================
# STOP CAMERA
# ============================================================

@app.route(
    "/api/camera/stop",
    methods=["POST"]
)
def api_camera_stop():

    global _latest_results

    with _results_lock:

        _latest_results = []

    release_camera()

    return jsonify({

        "success": True,

        "message":
            "Camera closed successfully."

    })


# ============================================================
# CAMERA STATUS
# ============================================================

@app.route("/api/camera/status")
def api_camera_status():

    camera_running = (

        _camera is not None

        and getattr(
            _camera,
            "is_running",
            False
        )

    )

    try:

        pipeline = get_pipeline()

        model_ready = getattr(
            getattr(pipeline, "recognizer", None),
            "is_trained",
            False
        )

    except Exception:

        model_ready = False

    return jsonify({

        "camera_running":
            camera_running,

        "model_ready":
            model_ready,

    })


# ============================================================
# LIVE VIDEO STREAM
# ============================================================

def _gen_live_stream():

    global _latest_results

    cam = _camera

    if cam is None or not getattr(
        cam,
        "is_running",
        False
    ):

        logger.info(
            "Camera is not running."
        )

        return

    try:

        pipeline = get_pipeline()

    except Exception as exc:

        logger.error(
            "Pipeline loading failed: %s",
            exc
        )

        return

    frame_counter = 0

    last_display_frame = None

    logger.info(
        "Live recognition stream started."
    )

    while getattr(
        cam,
        "is_running",
        False
    ):

        frame = cam.read()

        if frame is None:

            time.sleep(0.01)

            continue

        frame_counter += 1

        should_process = (

            frame_counter % FRAME_SKIP == 0

            or last_display_frame is None

        )

        if should_process:

            try:

                small_frame, scale = (
                    resize_for_processing(frame)
                )

                if small_frame is None:
                    continue

                with _live_processing_lock:

                    processed = pipeline.process_frame(
                        small_frame
                    )

                # Safely get annotated frame and results
                if (
                    isinstance(processed, tuple)
                    and len(processed) >= 2
                ):

                    annotated_small = processed[0]

                    results = processed[1]

                else:

                    annotated_small = small_frame

                    results = []

                # Prevent None frame
                if annotated_small is None:

                    annotated_small = small_frame

                if scale != 1.0:

                    annotated = cv2.resize(

                        annotated_small,

                        (
                            frame.shape[1],
                            frame.shape[0]
                        ),

                        interpolation=cv2.INTER_LINEAR

                    )

                else:

                    annotated = annotated_small

                last_display_frame = annotated

                normalized = normalize_results(
                    results
                )

                with _results_lock:

                    _latest_results = normalized

            except ModelNotTrainedError:

                last_display_frame = frame.copy()

                with _results_lock:

                    _latest_results = []

            except Exception as exc:

                logger.exception(
                    "Recognition error: %s",
                    exc
                )

                last_display_frame = frame.copy()

                with _results_lock:

                    _latest_results = []

        display_frame = (

            last_display_frame

            if last_display_frame is not None

            else frame

        )

        ok, buffer = cv2.imencode(

            ".jpg",

            display_frame,

            [

                cv2.IMWRITE_JPEG_QUALITY,

                JPEG_QUALITY

            ]

        )

        if not ok:

            continue

        yield (

            b"--frame\r\n"

            b"Content-Type: image/jpeg\r\n\r\n"

            + buffer.tobytes()

            + b"\r\n"

        )

    logger.info(
        "Live recognition stream ended."
    )


# ============================================================
# LIVE VIDEO FEED
# ============================================================

@app.route("/video_feed/live")
def video_feed_live():

    return Response(

        _gen_live_stream(),

        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),

    )


# ============================================================
# LIVE RESULTS API
# ============================================================

@app.route("/api/live/results")
def api_live_results():

    with _results_lock:

        results = normalize_results(
            _latest_results
        )

    return jsonify({

        "success": True,

        "results": make_json_safe(results)

    })


# ============================================================
# BROWSER CAMERA RECOGNITION
# ============================================================

@app.route(
    "/api/recognize-frame",
    methods=["POST"]
)
def recognize_frame():

    try:

        # ----------------------------------------------------
        # CHECK FRAME
        # ----------------------------------------------------

        if "frame" not in request.files:

            return jsonify({

                "success": False,

                "message":
                    "No frame received from browser.",

                "results": []

            }), 400

        image_file = request.files["frame"]

        if image_file is None:

            return jsonify({

                "success": False,

                "message":
                    "Invalid camera frame.",

                "results": []

            }), 400

        image_bytes = image_file.read()

        if not image_bytes:

            return jsonify({

                "success": False,

                "message":
                    "Empty camera frame.",

                "results": []

            }), 400

        # ----------------------------------------------------
        # DECODE IMAGE
        # ----------------------------------------------------

        np_array = np.frombuffer(
            image_bytes,
            np.uint8
        )

        if np_array is None or len(np_array) == 0:

            return jsonify({

                "success": False,

                "message":
                    "Invalid image data.",

                "results": []

            }), 400

        frame = cv2.imdecode(
            np_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            return jsonify({

                "success": False,

                "message":
                    "Invalid camera frame.",

                "results": []

            }), 400

        with _browser_liveness_lock:
            eyes_closed = detect_browser_blink(frame)
            if eyes_closed is False:
                if _browser_liveness["eyes_were_open"]:
                    _browser_liveness["blink_count"] += 1
                    _browser_liveness["valid_until"] = time.time() + 5.0
                _browser_liveness["eyes_were_open"] = True
            elif eyes_closed is True and _browser_liveness["eyes_were_open"]:
                _browser_liveness["eyes_were_open"] = False

            liveness_valid = (
                _browser_liveness["blink_count"]
                >= config.LIVENESS_REQUIRED_BLINKS
                and time.time() <= _browser_liveness["valid_until"]
            )

        if not liveness_valid:
            return jsonify({
                "success": True,
                "live": False,
                "message": "Blink eyes for detection.",
                "results": [],
            }), 200

        # ----------------------------------------------------
        # LOAD PIPELINE
        # ----------------------------------------------------

        pipeline = get_pipeline()

        if pipeline is None:

            return jsonify({

                "success": False,

                "message":
                    "Recognition pipeline is not available.",

                "results": []

            }), 500

        # ----------------------------------------------------
        # PROCESS FRAME
        # ----------------------------------------------------

        with _live_processing_lock:

            processed = pipeline.process_frame(
                frame
            )

        # ----------------------------------------------------
        # SAFELY EXTRACT RESULTS
        # ----------------------------------------------------

        results = extract_pipeline_results(
            processed
        )

        # Make absolutely sure it is a list
        results = normalize_results(
            results
        )

        # Make JSON safe
        results = make_json_safe(
            results
        )

        logger.debug(
            "Recognition results: %s",
            results
        )

        # ----------------------------------------------------
        # SUCCESS RESPONSE
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "results": results

        }), 200


    except ModelNotTrainedError:

        logger.warning(
            "Recognition attempted before model training."
        )

        return jsonify({

            "success": False,

            "message":
                "Model is not trained. "
                "Please train the model first.",

            "results": []

        }), 400


    except Exception as exc:

        # IMPORTANT:
        # This prevents the server from crashing
        # and always returns results as a list.

        logger.exception(
            "Browser recognition error: %s",
            exc
        )

        return jsonify({

            "success": False,

            "message": str(exc),

            "results": []

        }), 500


# ============================================================
# ATTENDANCE HISTORY
# ============================================================

@app.route("/attendance")
def attendance_history():

    date_filter = (

        request.args.get(
            "date",
            ""
        ).strip()

        or None

    )

    student_filter = (

        request.args.get(
            "student_id",
            ""
        ).strip().upper()

        or None

    )

    status_filter = (

        request.args.get(
            "status",
            ""
        ).strip()

        or None

    )

    records = database.get_attendance(

        date_str=date_filter,

        student_id=student_filter,

        status=status_filter,

    )

    all_students = (
        database.get_all_students()
    )

    return render_template(

        "attendance.html",

        history_mode=True,

        records=records,

        all_students=all_students,

        filters={

            "date":
                date_filter or "",

            "student_id":
                student_filter or "",

            "status":
                status_filter or "",

        },

    )


# ============================================================
# ATTENDANCE DATAFRAME
# ============================================================

def _attendance_dataframe(

    date_str=None,

    student_id=None

):

    records = database.get_attendance(

        date_str=date_str,

        student_id=student_id

    )

    df = pd.DataFrame(records)

    if df.empty:

        df = pd.DataFrame(

            columns=[

                "student_id",

                "name",

                "date",

                "time",

                "status",

                "confidence",

            ]

        )

    df = df.rename(

        columns={

            "student_id":
                "Student ID",

            "name":
                "Student Name",

            "date":
                "Date",

            "time":
                "Time",

            "status":
                "Status",

            "confidence":
                "Confidence",

        }

    )

    columns = [

        "Student ID",

        "Student Name",

        "Date",

        "Time",

        "Status",

        "Confidence",

    ]

    available_columns = [

        column

        for column in columns

        if column in df.columns

    ]

    return df[available_columns]


# ============================================================
# REPORTS PAGE
# ============================================================

@app.route("/reports")
def reports():

    date_filter = (

        request.args.get(
            "date",
            ""
        ).strip()

        or None

    )

    student_filter = (

        request.args.get(
            "student_id",
            ""
        ).strip().upper()

        or None

    )

    records = database.get_attendance(

        date_str=date_filter,

        student_id=student_filter

    )

    all_students = (
        database.get_all_students()
    )

    return render_template(

        "reports.html",

        records=records,

        all_students=all_students,

        filters={

            "date":
                date_filter or "",

            "student_id":
                student_filter or "",

        }

    )


# ============================================================
# EXPORT CSV
# ============================================================

@app.route("/reports/export.csv")
def export_csv():

    date_str = (
        request.args.get("date")
        or None
    )

    student_id = (
        request.args.get("student_id")
        or None
    )

    df = _attendance_dataframe(

        date_str=date_str,

        student_id=student_id

    )

    os.makedirs(

        config.REPORTS_DIR,

        exist_ok=True

    )

    filename = (

        "attendance_report_"

        + datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        + ".csv"

    )

    out_path = os.path.join(

        config.REPORTS_DIR,

        filename

    )

    df.to_csv(

        out_path,

        index=False

    )

    return send_file(

        out_path,

        as_attachment=True,

        download_name=filename

    )


# ============================================================
# EXPORT EXCEL
# ============================================================

@app.route("/reports/export.xlsx")
def export_excel():

    date_str = (
        request.args.get("date")
        or None
    )

    student_id = (
        request.args.get("student_id")
        or None
    )

    df = _attendance_dataframe(

        date_str=date_str,

        student_id=student_id

    )

    stats = database.get_dashboard_stats()

    os.makedirs(

        config.REPORTS_DIR,

        exist_ok=True

    )

    filename = (

        "attendance_report_"

        + datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        + ".xlsx"

    )

    out_path = os.path.join(

        config.REPORTS_DIR,

        filename

    )

    with pd.ExcelWriter(

        out_path,

        engine="xlsxwriter"

    ) as writer:

        df.to_excel(

            writer,

            sheet_name="Attendance",

            index=False

        )

        summary = pd.DataFrame([

            {

                "Metric":
                    "Total Students",

                "Value":
                    stats.get(
                        "total_students",
                        0
                    )

            },

            {

                "Metric":
                    "Present Today",

                "Value":
                    stats.get(
                        "present_today",
                        0
                    )

            },

            {

                "Metric":
                    "Absent Today",

                "Value":
                    stats.get(
                        "absent_today",
                        0
                    )

            },

            {

                "Metric":
                    "Attendance Percentage",

                "Value":
                    stats.get(
                        "attendance_percentage",
                        0
                    )

            },

        ])

        summary.to_excel(

            writer,

            sheet_name="Summary",

            index=False

        )

    return send_file(

        out_path,

        as_attachment=True,

        download_name=filename

    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    logger.info(
        "Starting Flask Face Attendance System..."
    )

    try:

        app.run(

            host=config.FLASK_HOST,

            port=config.FLASK_PORT,

            debug=config.FLASK_DEBUG,

            threaded=True,

            use_reloader=False,

        )

    finally:

        release_camera()