"""
scripts/extract_features.py
----------------------------
Reads every image in dataset/students/<ID>_<Name>/*.jpg, extracts a
VGGFace embedding for each (via app.face_recognizer.FaceEmbedder -- the
SAME class used at real-time recognition time, so there is exactly one
feature-extraction code path in the whole project) and saves the
resulting embeddings + labels to disk so scripts/train_model.py does not
need to re-run the (comparatively slow) VGGFace forward pass every time
you retrain the SVM.

Output:
    models/embeddings.npy   -- shape (N, embedding_dim), float32
    models/labels.npy       -- shape (N,), each entry is a student_id string

Usage:
    python scripts/extract_features.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from tqdm import tqdm

from app import config
from app.face_recognizer import FaceEmbedder
from app.utils import get_logger, EmptyDatasetError, CorruptedImageError

logger = get_logger(__name__)

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _parse_student_id_from_folder(folder_name: str) -> str:
    """Folder names look like 'ST001_John_Doe' -> student_id 'ST001'."""
    return folder_name.split("_", 1)[0]


def load_dataset_paths():
    """Return list of (image_path, student_id) tuples for every valid
    image under dataset/students/."""
    if not os.path.isdir(config.DATASET_PATH):
        raise EmptyDatasetError(f"Dataset directory not found: {config.DATASET_PATH}")

    pairs = []
    student_folders = [
        f for f in sorted(os.listdir(config.DATASET_PATH))
        if os.path.isdir(os.path.join(config.DATASET_PATH, f))
    ]
    if not student_folders:
        raise EmptyDatasetError(
            "No student folders found in dataset/students/. "
            "Register at least 2 students and collect face images first "
            "(python scripts/collect_faces.py or the web /students/register page)."
        )

    for folder in student_folders:
        student_id = _parse_student_id_from_folder(folder)
        folder_path = os.path.join(config.DATASET_PATH, folder)
        images = [f for f in sorted(os.listdir(folder_path)) if f.lower().endswith(VALID_EXTENSIONS)]
        if not images:
            logger.warning("Skipping %s: no images found.", folder)
            continue
        for img_name in images:
            pairs.append((os.path.join(folder_path, img_name), student_id))

    if not pairs:
        raise EmptyDatasetError("No valid images found in any student folder.")

    return pairs


def main() -> int:
    logger.info("Scanning dataset at %s ...", config.DATASET_PATH)
    try:
        pairs = load_dataset_paths()
    except EmptyDatasetError as exc:
        logger.error(str(exc))
        return 1

    unique_students = sorted(set(sid for _, sid in pairs))
    logger.info("Found %d images across %d students.", len(pairs), len(unique_students))

    if len(unique_students) < 2:
        logger.error(
            "At least 2 different students are required to train a classifier "
            "(found %d). Register more students before extracting features.",
            len(unique_students),
        )
        return 1

    cached = {}
    if os.path.isfile(config.EMBEDDING_CACHE_PATH):
        try:
            with open(config.EMBEDDING_CACHE_PATH, "r", encoding="utf-8") as cache_file:
                cache_data = json.load(cache_file)
            if cache_data.get("model_name") == config.VGGFACE_MODEL_NAME:
                cached = cache_data.get("images", {})
        except (OSError, ValueError) as exc:
            logger.warning("Ignoring invalid embedding cache: %s", exc)

    embeddings = []
    labels = []
    updated_cache = {}
    pending = []
    corrupted = 0

    for path, student_id in pairs:
        stat = os.stat(path)
        signature = f"{stat.st_mtime_ns}:{stat.st_size}"
        cached_entry = cached.get(path)
        if cached_entry and cached_entry.get("signature") == signature:
            updated_cache[path] = cached_entry
        else:
            pending.append((path, student_id, signature))

    logger.info(
        "Using %d cached embeddings; extracting %d new or changed images.",
        len(updated_cache),
        len(pending),
    )

    # Loading TensorFlow/DeepFace is expensive. Existing cached embeddings
    # should be enough when the dataset has not changed.
    embedder = FaceEmbedder() if pending else None

    for path, student_id, signature in tqdm(pending, desc="Extracting new features"):
        image = cv2.imread(path)
        if image is None:
            logger.warning("Could not decode image, skipping: %s", path)
            corrupted += 1
            continue
        try:
            embedding = embedder.embed(image)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to extract embedding for %s: %s", path, exc)
            corrupted += 1
            continue
        updated_cache[path] = {
            "student_id": student_id,
            "signature": signature,
            "embedding": embedding.tolist(),
        }

    if corrupted:
        logger.warning("%d image(s) were skipped due to read/embedding errors.", corrupted)

    for path, student_id in pairs:
        entry = updated_cache.get(path)
        if entry is None:
            continue
        embeddings.append(np.asarray(entry["embedding"], dtype=np.float32))
        labels.append(student_id)

    if not embeddings:
        logger.error("No embeddings could be extracted. Aborting.")
        return 1

    embeddings_arr = np.stack(embeddings, axis=0).astype("float32")
    labels_arr = np.array(labels)

    np.save(config.EMBEDDINGS_PATH, embeddings_arr)
    np.save(config.LABELS_PATH, labels_arr)
    with open(config.EMBEDDING_CACHE_PATH, "w", encoding="utf-8") as cache_file:
        json.dump(
            {
                "model_name": config.VGGFACE_MODEL_NAME,
                "images": updated_cache,
            },
            cache_file,
        )

    logger.info("Saved %s (%s)", config.EMBEDDINGS_PATH, embeddings_arr.shape)
    logger.info("Saved %s (%s)", config.LABELS_PATH, labels_arr.shape)
    logger.info("Next step: python scripts/train_model.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
