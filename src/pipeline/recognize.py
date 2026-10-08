from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from ultralytics import YOLO

ROOT = Path(r"D:\internship_RIED\ai-model")
DB = ROOT / "dataset" / "biometric" / "database"

MODEL_VALIDATOR = ROOT / "models" / "nose_validator" / "best_phase1.keras"
MODEL_BIOMETRIC = ROOT / "models" / "biometric" / "best_biometric.keras"
MODEL_YOLO      = ROOT / "models" / "nose_detector" / "yolo11n_nose_tight" / "weights" / "best.pt"

IMG_SIZE = (224, 224)

# A pre-cropped nose is usually small. A full dog photo is large.
# If YOLO fails AND image is large, we don't fall back to the full image.
SMALL_IMAGE_MAX_SIDE = 500

VALIDATOR_THRESHOLD = 0.5
MATCH_THRESHOLD     = 0.80

_validator = None
_biometric = None
_yolo = None


def _load_all():
    global _validator, _biometric, _yolo
    if _validator is None:
        print("Loading validator...")
        _validator = tf.keras.models.load_model(MODEL_VALIDATOR, compile=False)
    if _biometric is None:
        print("Loading biometric model...")
        _biometric = tf.keras.models.load_model(MODEL_BIOMETRIC, compile=False)
    if _yolo is None:
        print("Loading YOLO detector...")
        _yolo = YOLO(str(MODEL_YOLO))


def _preprocess_bgr_to_rgb(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_rgb, IMG_SIZE)
    img = img.astype(np.float32)
    img = preprocess_input(img)
    return img


def _detect_nose(image_path, img_bgr):
    # Ultralytics works best when given a file path
    results = _yolo(str(image_path), conf=0.25, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    confs = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    i = int(np.argmax(confs))
    x1, y1, x2, y2 = xyxy[i]
    h, w = img_bgr.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if (x2 - x1) < 16 or (y2 - y1) < 16:
        return None
    return img_bgr[y1:y2, x1:x2]


def _validator_score(img_bgr):
    pre = _preprocess_bgr_to_rgb(img_bgr)
    return float(_validator.predict(tf.expand_dims(pre, 0), verbose=0)[0, 0])


def recognize(image_path, top_k=5):
    _load_all()

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return {"status": "error", "message": f"Could not read {image_path}"}

    h, w = img_bgr.shape[:2]
    is_small = max(h, w) <= SMALL_IMAGE_MAX_SIDE

    # ---- Candidate crops ----
    candidates = []

    yolo_crop = _detect_nose(image_path, img_bgr)
    if yolo_crop is not None and yolo_crop.size > 0:
        candidates.append(("yolo", yolo_crop))

    # Only fall back to full image if it looks like an already-cropped nose.
    if is_small:
        candidates.append(("full", img_bgr))

    if not candidates:
        return {
            "status": "not_a_nose",
            "message": "No nose detected and image is too large to be a crop.",
            "validator_score": 0.0,
            "used_yolo": False,
        }

    # ---- Pick best candidate by validator score ----
    best_source = None
    best_crop = None
    best_score = -1.0
    scores = {}
    for source, crop in candidates:
        s = _validator_score(crop)
        scores[source] = s
        if s > best_score:
            best_score = s
            best_source = source
            best_crop = crop

    used_detector = (best_source == "yolo")

    if best_score < VALIDATOR_THRESHOLD:
        return {
            "status": "not_a_nose",
            "message": f"Not a dog nose (best validator={best_score:.3f})",
            "validator_score": best_score,
            "validator_scores": scores,
            "used_yolo": used_detector,
        }

    # ---- Biometric embedding ----
    pre = _preprocess_bgr_to_rgb(best_crop)
    vec = _biometric.predict(tf.expand_dims(pre, 0), verbose=0)[0].astype(np.float32)

    # ---- Match against database ----
    db_emb = np.load(DB / "embeddings.npy")
    db_ids = np.load(DB / "dog_ids.npy")

    sims = db_emb @ vec
    top_idx = np.argsort(-sims)[:top_k]
    top_matches = [
        {"dog_id": int(db_ids[i]), "similarity": float(sims[i])}
        for i in top_idx
    ]
    best = top_matches[0]

    if best["similarity"] >= MATCH_THRESHOLD:
        return {
            "status": "matched",
            "dog_id": best["dog_id"],
            "confidence": best["similarity"],
            "validator_score": best_score,
            "validator_scores": scores,
            "used_yolo": used_detector,
            "top5": top_matches,
        }
    else:
        return {
            "status": "unknown",
            "message": "No matching dog. Register as new?",
            "best_guess": best,
            "validator_score": best_score,
            "validator_scores": scores,
            "used_yolo": used_detector,
            "top5": top_matches,
        }