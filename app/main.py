"""
app/main.py
-----------

Flask application for the Face Attendance System.

Features:
- Student registration
- Face image capture
- Model training
- Live face recognition
- Multiple face recognition
- Attendance marking
- Attendance history
- CSV and Excel reports
"""

import atexit
import os
import subprocess
import sys
import threading
import time
from datetime import datetime

import cv2
import pandas as pd
import numpy as np

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
    Response,
    send_file,
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


# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

FRAME_SKIP = 2
MAX_PROCESS_WIDTH = 960
JPEG_QUALITY = 80


# ============================================================
# CAMERA FUNCTIONS
# ============================================================

def get_camera():
    global _camera

    # Reuse existing camera instead of creating it again
    if _camera is not None:
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
# RESIZE FRAME FOR AI PROCESSING
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
def not_found(_error):

    return render_template(
        "error.html",
        message="Page not found."
    ), 404


@app.errorhandler(500)
def server_error(error):
    logger.error("Server error: %s", error)

    return render_template(
        "error.html",
        message="An unexpected error occurred."
    ), 500


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
def index():

    stats = database.get_dashboard_stats()

    try:
        model_ready = get_pipeline().recognizer.is_trained

    except Exception:
        model_ready = False

    return render_template(
        "index.html",
        stats=stats,
        model_ready=model_ready,
    )


@app.route("/dashboard")
def dashboard():

    stats = database.get_dashboard_stats()

    recent = database.get_attendance()[:10]

    camera_running = (
        _camera is not None
        and _camera.is_running
    )

    try:
        model_ready = get_pipeline().recognizer.is_trained

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


@app.route(
    "/students/delete/<student_id>",
    methods=["POST"]
)
def delete_student(student_id):

    database.delete_student(student_id)

    flash(
        f"Student {student_id} deleted successfully.",
        "warning"
    )

    return redirect(
        url_for("students")
    )


# ============================================================
# REGISTER STUDENT
# ============================================================

@app.route(
    "/students/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "POST":

        student_id = sanitize_student_id(
            request.form.get("student_id", "")
        )

        name = sanitize_name(
            request.form.get("name", "")
        )

        department = request.form.get(
            "department",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip()

        if not student_id or not name:

            flash(
                "Student ID and Name are required.",
                "danger"
            )

            return render_template(
                "register.html"
            )

        if database.student_exists(student_id):

            flash(
                f"Student ID {student_id} already exists.",
                "danger"
            )

            return render_template(
                "register.html"
            )

        folder_name = f"{student_id}_{name}"

        folder_path = os.path.join(
            config.DATASET_PATH,
            folder_name
        )

        os.makedirs(
            folder_path,
            exist_ok=True
        )

        added = database.add_student(
            student_id,
            name.replace("_", " "),
            department,
            email
        )

        if not added:

            flash(
                "Could not register student.",
                "danger"
            )

            return render_template(
                "register.html"
            )

        flash(
            "Student registered successfully. Capture face images now.",
            "success"
        )

        return redirect(
            url_for(
                "capture",
                student_id=student_id,
                name=name
            )
        )

    return render_template(
        "register.html"
    )


# ============================================================
# CAPTURE PAGE
# ============================================================

@app.route(
    "/students/register/capture/<student_id>"
)
def capture(student_id):

    name = request.args.get(
        "name",
        student_id
    )

    student = database.get_student(
        student_id
    )

    if student is None:

        flash(
            "Unknown student.",
            "danger"
        )

        return redirect(
            url_for("register")
        )

    folder_name = f"{student_id}_{name}"

    folder_path = os.path.join(
        config.DATASET_PATH,
        folder_name
    )

    existing = 0

    if os.path.isdir(folder_path):

        existing = len([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ])

    return render_template(
        "register.html",
        capture_mode=True,
        student_id=student_id,
        name=name,
        folder_name=folder_name,
        existing_count=existing,
        target_count=config.NUM_COLLECTION_IMAGES,
    )


# ============================================================
# REGISTRATION CAMERA STREAM
# ============================================================

def _gen_registration_stream(student_id, folder_name):

    try:
        cam = get_camera()

        if not cam.is_running:
            cam.start()

    except CameraUnavailableError as exc:

        logger.error(
            "Camera unavailable: %s",
            exc
        )

        return

    while cam.is_running:

        frame = cam.read()

        if frame is None:

            time.sleep(0.01)
            continue

        annotated = frame.copy()

        try:

            face = _registration_detector.detect_largest(
                frame
            )

        except Exception as exc:

            logger.warning(
                "Face detection error: %s",
                exc
            )

            face = None

        if face is not None:

            x, y, w, h = face.box

            cv2.rectangle(
                annotated,
                (x, y),
                (x + w, y + h),
                (0, 200, 0),
                2
            )

            cv2.putText(
                annotated,
                "Face detected",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 0),
                2
            )

        else:

            cv2.putText(
                annotated,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        ok, buffer = cv2.imencode(
            ".jpg",
            annotated,
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

@app.route(
    "/video_feed/register/<student_id>"
)
def video_feed_register(student_id):

    folder_name = request.args.get(
        "folder",
        student_id
    )

    return Response(
        _gen_registration_stream(
            student_id,
            folder_name
        ),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),
    )


# ============================================================
# CAPTURE FACE IMAGE
# ============================================================

@app.route(
    "/api/capture/<student_id>",
    methods=["POST"]
)
def api_capture(student_id):

    data = request.get_json(
        silent=True
    ) or {}

    folder_name = data.get("folder_name")

    if not folder_name:

        return jsonify({
            "success": False,
            "message": "Dataset folder missing."
        }), 400

    folder_path = os.path.join(
        config.DATASET_PATH,
        folder_name
    )

    os.makedirs(
        folder_path,
        exist_ok=True
    )

    cam = get_camera()

    if not cam.is_running:

        try:
            cam.start()

        except CameraUnavailableError as exc:

            return jsonify({
                "success": False,
                "message": str(exc)
            }), 503

    frame = cam.read()

    if frame is None:

        return jsonify({
            "success": False,
            "message": "Camera not ready. Please wait."
        }), 503

    face = _registration_detector.detect_largest(
        frame
    )

    if face is None:

        return jsonify({
            "success": False,
            "message": "No face detected. Please face the camera."
        })

    crop = _registration_detector.crop_face(
        frame,
        face
    )

    if crop is None:

        return jsonify({
            "success": False,
            "message": "Invalid face crop."
        })

    crop_resized = cv2.resize(
        crop,
        config.IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    existing = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(".jpg")
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
            "message": "Could not save image."
        }), 500

    done = (
        next_index >= config.NUM_COLLECTION_IMAGES
    )

    return jsonify({

        "success": True,

        "count": next_index,

        "target": config.NUM_COLLECTION_IMAGES,

        "done": done,

        "message":
            f"Captured {next_index}/"
            f"{config.NUM_COLLECTION_IMAGES}",

    })


# ============================================================
# MODEL TRAINING
# ============================================================

@app.route(
    "/api/train",
    methods=["POST"]
)
def api_train():

    global _pipeline

    release_camera()

    python_exe = sys.executable

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
                    extract.stderr[-4000:]

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
                    train.stderr[-4000:]

            }), 500

    except subprocess.TimeoutExpired:

        return jsonify({

            "success": False,

            "message":
                "Training timed out."

        }), 500

    except Exception as exc:

        logger.exception(
            "Training failed"
        )

        return jsonify({

            "success": False,

            "message":
                str(exc)

        }), 500

    _pipeline = None

    return jsonify({

        "success": True,

        "log":
            extract.stdout[-2000:]
            + "\n"
            + train.stdout[-2000:]

    })


# ============================================================
# LIVE ATTENDANCE PAGE
# ============================================================

@app.route("/attendance/live")
def live_attendance():

    try:

        model_ready = (
            get_pipeline()
            .recognizer
            .is_trained
        )

    except Exception:

        model_ready = False

    camera_running = (
        _camera is not None
        and _camera.is_running
    )

    return render_template(

        "attendance.html",

        model_ready=model_ready,

        camera_running=camera_running,

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

        if not cam.is_running:
            cam.start()

        return jsonify({
            "success": True
        })

    except CameraUnavailableError as exc:

        logger.error(
            "Camera unavailable: %s",
            exc
        )

        return jsonify({

            "success": False,

            "message":
                str(exc)

        }), 503

    except Exception as exc:

        logger.exception(
            "Camera start error"
        )

        return jsonify({

            "success": False,

            "message":
                str(exc)

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

    # Release physical webcam
    release_camera()

    logger.info(
        "Camera closed by user."
    )

    return jsonify({
        "success": True,
        "message": "Camera closed successfully."
    })


# ============================================================
# CAMERA STATUS
# ============================================================

@app.route("/api/camera/status")
def api_camera_status():

    camera_running = (
        _camera is not None
        and _camera.is_running
    )

    try:

        model_ready = (
            get_pipeline()
            .recognizer
            .is_trained
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

    # Do NOT create or start the camera here.
    # Camera must be started explicitly using /api/camera/start.

    cam = _camera

    if cam is None or not cam.is_running:

        logger.info(
            "Live stream requested but camera is not running."
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

    while cam.is_running:

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

                small_frame, scale = resize_for_processing(
                    frame
                )

                with _live_processing_lock:

                    annotated_small, results = (
                        pipeline.process_frame(
                            small_frame
                        )
                    )

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


                with _results_lock:

                    _latest_results = results or []


            except ModelNotTrainedError:

                last_display_frame = frame.copy()

                with _results_lock:

                    _latest_results = []


            except Exception as exc:

                logger.warning(
                    "Recognition error: %s",
                    exc
                )

                last_display_frame = frame.copy()


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

        results = list(
            _latest_results
        )

    return jsonify({
        "results": results
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

        if "image" not in request.files:

            return jsonify({

                "success": False,

                "message": "No image received."

            }), 400


        image_file = request.files["image"]

        image_bytes = image_file.read()


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

                "message": "Invalid image."

            }), 400


        pipeline = get_pipeline()


        annotated_frame, results = (
            pipeline.process_frame(frame)
        )


        return jsonify({

            "success": True,

            "results": results or []

        })


    except Exception as exc:

        logger.exception(
            "Browser recognition error"
        )


        return jsonify({

            "success": False,

            "message": str(exc)

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

    all_students = database.get_all_students()

    return render_template(
        "reports.html",
        records=records,
        all_students=all_students,
        filters={
            "date": date_filter or "",
            "student_id": student_filter or "",
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