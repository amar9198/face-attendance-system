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
    "/api/capture/<student_id>",
    methods=["POST"]
)
def api_capture(student_id):

    try:

        # ----------------------------------------------------
        # CHECK STUDENT
        # ----------------------------------------------------

        student = database.get_student(student_id)

        logger.info(
            "Capture request received for student_id=%s, student=%s",
            student_id,
            student
        )

        if student is None:

            logger.warning(
                "Student not found during capture: %s",
                student_id
            )

            return jsonify({
                "success": False,
                "message": (
                    f"Student {student_id} not found. "
                    "Please register the student again."
                )
            }), 404


        # ----------------------------------------------------
        # GET FOLDER NAME
        # ----------------------------------------------------

        folder_name = request.form.get("folder_name")

        if not folder_name:

            student_name = student.get("name", "Unknown")

            folder_name = (
                f"{student_id}_"
                f"{student_name.replace(' ', '_')}"
            )


        # ----------------------------------------------------
        # CREATE DATASET FOLDER
        # ----------------------------------------------------

        folder_path = os.path.join(
            config.DATASET_PATH,
            folder_name
        )

        os.makedirs(
            folder_path,
            exist_ok=True
        )


        # ----------------------------------------------------
        # GET IMAGE FROM BROWSER
        # ----------------------------------------------------

        if "image" not in request.files:

            logger.warning(
                "No image received for student: %s",
                student_id
            )

            return jsonify({
                "success": False,
                "message": "No image received from camera."
            }), 400


        image_file = request.files["image"]

        image_bytes = image_file.read()


        # ----------------------------------------------------
        # CONVERT IMAGE TO OPENCV FRAME
        # ----------------------------------------------------

        np_array = np.frombuffer(
            image_bytes,
            np.uint8
        )

        frame = cv2.imdecode(
            np_array,
            cv2.IMREAD_COLOR
        )


        if frame is None:

            logger.warning(
                "Invalid camera image for student: %s",
                student_id
            )

            return jsonify({
                "success": False,
                "message": "Invalid camera image."
            }), 400


        # ----------------------------------------------------
        # DETECT FACE
        # ----------------------------------------------------

        logger.info(
            "Detecting face for student: %s",
            student_id
        )

        face = _registration_detector.detect_largest(
            frame
        )


        if face is None:

            return jsonify({
                "success": False,
                "message": (
                    "No face detected. "
                    "Please look directly at the camera."
                )
            })


        # ----------------------------------------------------
        # CROP FACE
        # ----------------------------------------------------

        crop = _registration_detector.crop_face(
            frame,
            face
        )


        if crop is None:

            return jsonify({
                "success": False,
                "message": "Could not crop detected face."
            })


        # ----------------------------------------------------
        # RESIZE FACE
        # ----------------------------------------------------

        crop_resized = cv2.resize(
            crop,
            config.IMAGE_SIZE,
            interpolation=cv2.INTER_AREA
        )


        # ----------------------------------------------------
        # COUNT EXISTING IMAGES
        # ----------------------------------------------------

        existing = [

            f for f in os.listdir(folder_path)

            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )

        ]


        next_index = len(existing) + 1


        # ----------------------------------------------------
        # SAVE IMAGE
        # ----------------------------------------------------

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

            logger.error(
                "Could not save image: %s",
                output_path
            )

            return jsonify({
                "success": False,
                "message": "Could not save captured image."
            }), 500


        # ----------------------------------------------------
        # CHECK COMPLETION
        # ----------------------------------------------------

        done = (
            next_index >= config.NUM_COLLECTION_IMAGES
        )


        logger.info(
            "Image captured successfully: %s (%s/%s)",
            student_id,
            next_index,
            config.NUM_COLLECTION_IMAGES
        )


        return jsonify({

            "success": True,

            "count": next_index,

            "target": config.NUM_COLLECTION_IMAGES,

            "done": done,

            "message": (
                f"Captured {next_index}/"
                f"{config.NUM_COLLECTION_IMAGES}"
            )

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

# ============================================================
# BROWSER CAMERA RECOGNITION
# ============================================================

@app.route(
    "/api/recognize-frame",
    methods=["POST"]
)
def recognize_frame():

    try:

        # Accept the field sent by attendance.html
        if "frame" not in request.files:

            return jsonify({
                "success": False,
                "message": "No frame received from browser."
            }), 400


        image_file = request.files["frame"]

        image_bytes = image_file.read()


        # Convert bytes to OpenCV image
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
                "message": "Invalid camera frame."
            }), 400


        # Load recognition pipeline
        pipeline = get_pipeline()


        # Process frame
        annotated_frame, results = (
            pipeline.process_frame(frame)
        )


        return jsonify({

            "success": True,

            "results": results or []

        })


    except ModelNotTrainedError:

        return jsonify({

            "success": False,

            "message":
                "Model is not trained. Please train the model first."

        }), 400


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