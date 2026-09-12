# Real-Time Face Recognition Attendance System

A complete, runnable attendance management system that detects and recognizes
students from a live webcam feed and automatically records their attendance.

**Pipeline:** `Webcam → MTCNN → VGGFace → SVM → Student Identification → Attendance Database → Dashboard/Reports`

---

## 1. Project Description

This project implements the classic face-recognition attendance methodology:

1. **MTCNN** detects faces (possibly multiple) in each webcam frame.
2. **VGGFace** extracts a fixed-length embedding vector from each detected face.
3. An **SVM** classifier (scikit-learn) maps the embedding to a student identity,
   or rejects it as **Unknown** if the confidence is below a configurable threshold.
4. Recognized, known students are written to a **SQLite** attendance database,
   with duplicate-attendance prevention (a student is marked present only once
   per day, enforced both in application logic and by a database constraint).
5. A **Flask** web dashboard lets you register students, run live recognition,
   browse attendance history, and export **CSV/Excel** reports.

## 2. Features

- Student registration (web UI **and** CLI script) with guided webcam face capture.
- MTCNN multi-face detection with a configurable confidence threshold.
- Reusable VGGFace feature-extraction class shared by training and real-time recognition.
- SVM multi-class classifier with an explicit "Unknown" rejection threshold.
- Live webcam recognition page with bounding boxes, names, IDs, status and confidence.
- Duplicate-attendance prevention at both the application and database level.
- Attendance history with filtering by date / student / status.
- CSV and Excel report export with summary statistics.
- Model evaluation script (accuracy, precision, recall, F1, confusion matrix) on a
  held-out test split that the model never saw during training.
- Centralized configuration (`app/config.py`) — no hard-coded values scattered
  around the codebase.
- Defensive error handling for missing camera, missing model, empty dataset,
  corrupted images, unknown faces, etc.

## 3. Architecture

```
face-attendance-system/
├── app/
│   ├── __init__.py
│   ├── main.py            Flask application (routes, video streaming, reports)
│   ├── config.py          All configurable constants
│   ├── camera.py          Webcam capture thread + real-time recognition pipeline
│   ├── face_detector.py   MTCNN wrapper (DetectedFace, FaceDetector)
│   ├── face_recognizer.py VGGFace embedding extractor + SVM classifier wrapper
│   ├── attendance.py      Attendance business logic / duplicate prevention
│   ├── database.py        SQLite schema + data access layer
│   └── utils.py           Logging, custom exceptions, shared helpers
│
├── scripts/
│   ├── test_camera.py       Step 1: webcam + MTCNN sanity check
│   ├── collect_faces.py     CLI student registration + face capture
│   ├── extract_features.py Batch VGGFace embedding extraction
│   ├── train_model.py       SVM training (train/val/test split)
│   └── evaluate_model.py    Accuracy / precision / recall / F1 / confusion matrix
│
├── dataset/students/        <StudentID>_<Name>/*.jpg   (captured face images)
├── models/                  svm_model.pkl, label_encoder.pkl, embeddings.npy, ...
├── attendance/attendance.db SQLite database (auto-created)
├── reports/                 Generated CSV/Excel exports
├── templates/                Flask/Jinja2 HTML templates
├── static/css, static/js    Styling and frontend JS
├── requirements.txt
└── README.md
```

### VGGFace compatibility note (important)

The original `keras_vggface` package (rcmalli/keras-vggface) depends on the
standalone `keras` 2.2.x API bundled with TensorFlow 1.x, and **cannot be
installed on Python 3.10/3.11 with modern TensorFlow 2.x** — pip will fail to
resolve its dependencies.

Rather than replacing VGGFace with a different recognition architecture, this
project uses the `deepface` package's `"VGG-Face"` model, which is an
actively-maintained, TensorFlow-2/Keras-compatible re-implementation of the
**same VGG-16-based VGGFace architecture and pretrained weights** referenced
in the original methodology. The intended pipeline — **MTCNN → VGGFace → SVM**
— is unchanged; only the library providing the VGGFace forward pass differs.
This is implemented in a single place, `app/face_recognizer.py::FaceEmbedder`,
which both the training scripts and the real-time recognizer use, so there is
exactly one feature-extraction code path in the whole project.

## 4. Requirements

- Python 3.10 or 3.11 (TensorFlow 2.15 does not support 3.12 at the time of writing)
- A webcam
- ~2 GB free disk space (TensorFlow + model weights)
- Windows, macOS, or Linux

## 5. Installation

### 5.1 Clone / copy the project

```bash
cd face-attendance-system
```

### 5.2 Create a virtual environment

**Windows (PowerShell):**
```powershell
py -3.11 -m venv venv
venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 5.3 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> The first `deepface` VGG-Face call downloads ~500MB of pretrained weights
> to `~/.deepface/weights/` (Linux/macOS) or `%USERPROFILE%\.deepface\weights\`
> (Windows). This requires an internet connection the first time only.

## 6. Usage — Step by Step

### 6.1 Test the camera + MTCNN

```bash
python scripts/test_camera.py
```
A window should open showing your webcam feed with a green box around any
detected face and a live confidence percentage. Press `q` to quit.

### 6.2 Start the web application

```bash
python app/main.py
```
The database (`attendance/attendance.db`) is created automatically on
startup — no manual SQL required. Open your browser at:

```
http://localhost:5000
```

### 6.3 Register a student

**Option A — Web UI:** go to **Register Student**, fill in the details, tick
the consent checkbox, then use the on-screen **Capture Photo** button (with
live camera preview) to collect ~25 images at different angles/expressions.

**Option B — Command line:**
```bash
python scripts/collect_faces.py
```
Follow the prompts; press SPACE to capture each image, `q` to stop early.

Register **at least 2 different students** — an SVM classifier needs at
least 2 classes to be meaningful.

### 6.4 Extract features

```bash
python scripts/extract_features.py
```
Reads every image under `dataset/students/`, extracts a VGGFace embedding
for each with the same `FaceEmbedder` class used at recognition time, and
saves `models/embeddings.npy` + `models/labels.npy`.

### 6.5 Train the SVM

```bash
python scripts/train_model.py
```
Splits the embeddings into train/validation/test sets, trains an SVM,
and saves `models/svm_model.pkl`, `models/label_encoder.pkl`, and a
`models/test_split.npz` file used by the evaluation script.

> Tip: from the web UI, the **Students** page has a **Train / Update Model**
> button that runs steps 6.4 and 6.5 for you as background processes.

### 6.6 Evaluate the model

```bash
python scripts/evaluate_model.py
```
Prints Accuracy / Precision / Recall / F1-score and a confusion matrix,
computed on the held-out **test** split (never seen during training).

> These numbers reflect *your* dataset. They are not a guaranteed
> reproduction of any accuracy reported in a reference paper.

### 6.7 Live attendance

Go to **Live Attendance** in the web UI, click **Start Camera**. Each
detected face is boxed, labeled with name/ID/confidence, and marked
**Present** in green the first time it's recognized in the current day;
subsequent detections show **Already Present** in orange (no duplicate
database rows). Faces below the recognition threshold show **Unknown**
in red and are never marked present. Click **Stop Camera** to release
the webcam.

### 6.8 Attendance history & reports

- **Attendance History**: filterable table (date / student / status).
- **Reports**: export the (optionally filtered) attendance table as CSV
  or Excel, with a summary sheet (total/present/absent/percentage).

## 7. Command Reference

| Command | What it does |
|---|---|
| `python scripts/test_camera.py` | Webcam + MTCNN sanity check |
| `python scripts/collect_faces.py` | CLI: register a student + capture face images |
| `python scripts/extract_features.py` | Extract VGGFace embeddings for the whole dataset |
| `python scripts/train_model.py` | Train/validate the SVM classifier |
| `python scripts/evaluate_model.py` | Evaluate on the held-out test split |
| `python app/main.py` | Start the Flask web application |

## 8. Configuration

All tunable values live in `app/config.py`, several overridable via
environment variables:

| Setting | Purpose | Default |
|---|---|---|
| `CAMERA_INDEX` | Which webcam to use | `0` |
| `FACE_CONFIDENCE_THRESHOLD` | Minimum MTCNN detection confidence | `0.90` |
| `RECOGNITION_THRESHOLD` | Minimum SVM probability to accept an identity (else "Unknown") | `0.55` |
| `IMAGE_SIZE` | VGGFace input size | `(224, 224)` |
| `NUM_COLLECTION_IMAGES` | Images captured per student during registration | `25` |
| `DATASET_PATH` | Where face images are stored | `dataset/students` |
| `MODEL_PATH` (`SVM_MODEL_PATH` etc.) | Where trained model artifacts live | `models/` |
| `DATABASE_PATH` | SQLite file location | `attendance/attendance.db` |

Example (Linux/macOS):
```bash
CAMERA_INDEX=1 RECOGNITION_THRESHOLD=0.65 python app/main.py
```

## 9. Troubleshooting

- **Camera won't open / "Could not open camera"** — Make sure no other
  application (Zoom, Teams, another Python process) is using the webcam,
  and that OS camera permissions are granted to your terminal/IDE. Try a
  different `CAMERA_INDEX` (0, 1, 2...).
- **`pip install mtcnn` or `tensorflow` fails** — Confirm you're on Python
  3.10 or 3.11 (`python --version`), not 3.12+ or 3.9-.
- **`deepface` first run is very slow / hangs** — It's downloading model
  weights over the network the first time; ensure you have an internet
  connection and enough disk space, then retry.
- **"Model not trained" banner on Live Attendance** — Run
  `extract_features.py` then `train_model.py` (or the **Train / Update
  Model** button) after registering at least 2 students with images.
- **Everyone shows "Unknown"** — Lower `RECOGNITION_THRESHOLD` in
  `app/config.py`, collect more/varied training images per student, and
  ensure good, even lighting during both registration and recognition.
- **Attendance not being marked** — Check `logs/attendance_system.log` for
  errors; confirm the student's face was actually registered and the model
  was retrained after registration.
- **Duplicate students / SVM confuses two people** — Collect more images
  per student (30+) with varied angles and lighting, and consider lowering
  the SVM `C` value or checking `RECOGNITION_THRESHOLD`.
- **Report export fails** — Check that `openpyxl`/`XlsxWriter` installed
  correctly (`pip show xlsxwriter`).

## 10. Project Limitations

- Recognition accuracy depends heavily on the quantity/quality of
  registration images (lighting, angle variety, image count).
- A linear-kernel SVM is used for simplicity/interpretability; very large
  student populations may benefit from a different kernel or classifier.
- The system marks one "Present" record per student per day — it does not
  model multiple class sessions/periods out of the box (the schema can be
  extended with a `session_id` column for that).
- No authentication/authorization layer is implemented on the Flask app;
  do not expose it directly to the public internet without adding one.
- Real-time performance depends on your CPU/GPU; MTCNN + VGGFace on CPU-only
  machines may process only a few frames per second.

## 11. Future Improvements

- Add user authentication/roles (admin vs. teacher) to the Flask app.
- Support multiple camera feeds / classrooms simultaneously.
- Add a "session" concept (per class period) instead of once-per-day.
- GPU acceleration and/or a lighter-weight embedding model for edge devices.
- Automatic dataset augmentation (brightness/rotation) during collection.
- Email/SMS notifications for absentee students.

## 12. Privacy & Data Protection (Important)

This system captures and stores **biometric facial data**. Before deploying it:

- **Inform students clearly** that facial images will be captured and used
  for attendance, and obtain their **explicit, informed consent** (the web
  registration form includes a consent checkbox as a starting point — adapt
  it to your institution's legal requirements).
- Apply **data protection and retention policies**: define how long face
  images and embeddings are kept, and delete them when a student leaves.
- Restrict **access controls** to the `dataset/`, `models/`, and
  `attendance/` directories — they contain personally identifiable
  biometric data and attendance records.
- Comply with applicable local/regional data protection law (e.g. GDPR,
  FERPA, or your institution's own policy) regarding biometric data
  collection, processing, and storage.
- The `.gitignore` in this project already excludes `dataset/students/*`,
  model files, and the database from version control by default — do not
  remove those exclusions without a deliberate data-handling policy.

## 13. Acceptance Checklist

- [x] `pip install -r requirements.txt` installs cleanly on Python 3.10/3.11
- [x] `python app/main.py` starts the Flask app; SQLite tables auto-created
- [x] Register a student via web UI or `collect_faces.py`
- [x] Capture ~25 face images per student (MTCNN-validated before saving)
- [x] `extract_features.py` + `train_model.py` (or the Train button) build the model
- [x] Live Attendance detects/recognizes faces, draws boxes with name/ID/confidence
- [x] First recognition marks Present in SQLite; repeat recognitions do not duplicate
- [x] Unrecognized faces show "Unknown" and are never marked present
- [x] Attendance History displays and filters records
- [x] CSV/Excel reports can be exported
- [x] `evaluate_model.py` prints accuracy/precision/recall/F1 + confusion matrix
