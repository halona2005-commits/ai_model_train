"""
Single validated nose-preparation function used by BOTH
register.py and recognize.py.

Every path that enrolls or matches a nose MUST call this first.

Returns a structured result with status + reason codes.
"""

from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

MODEL_VALIDATOR = ROOT / "models" / "nose_validator" / "best_phase1.keras"
MODEL_YOLO      = ROOT / "models" / "nose_detector" / "yolo11n_nose_tight" / "weights" / "best.pt"

IMG_SIZE = (224, 224)
SMALL_IMAGE_MAX_SIDE = 500
VALIDATOR_THRESHOLD  = 0.5

_validator = None
_yolo      = None


def _load():
    global _validator, _yolo
    if _validator is None:
        _validator = tf.keras.models.load_model(MODEL_VALIDATOR, compile=False)
    if _yolo is None:
        _yolo = YOLO(str(MODEL_YOLO))


def _preprocess_bgr_to_rgb(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_rgb, IMG_SIZE)
    img = img.astype(np.float32)
    img = preprocess_input(img)
    return img


def _validator_score(img_bgr):
    pre = _preprocess_bgr_to_rgb(img_bgr)
    return float(_validator.predict(tf.expand_dims(pre, 0), verbose=0)[0, 0])


def _yolo_crop(image_path, img_bgr):
    yolo_input = str(image_path) if image_path is not None else img_bgr
    results = _yolo(yolo_input, conf=0.25, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    i = int(np.argmax(confs))
    x1, y1, x2, y2 = xyxy[i]
    h, w = img_bgr.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if (x2 - x1) < 16 or (y2 - y1) < 16:
        return None
    return img_bgr[y1:y2, x1:x2]


def prepare_nose(image_path=None, img_bgr=None):
    """
    Single source of truth for nose preparation.

    Returns dict:
        {
            "status": "accepted" | "rejected",
            "reason_code": "OK" | "IMAGE_UNREADABLE" | "NO_NOSE_DETECTED"
                           | "VALIDATOR_REJECTED" | "CROP_TOO_SMALL",
            "crop_bgr": ndarray or None,    # only if accepted
            "validator_score": float,
            "used_yolo": bool,
            "validator_scores": dict,
        }
    """
    _load()

    if img_bgr is None:
        if image_path is None:
            return {"status": "rejected", "reason_code": "IMAGE_UNREADABLE",
                    "crop_bgr": None, "validator_score": 0.0}
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return {"status": "rejected", "reason_code": "IMAGE_UNREADABLE",
                    "crop_bgr": None, "validator_score": 0.0}

    h, w = img_bgr.shape[:2]
    is_small = max(h, w) <= SMALL_IMAGE_MAX_SIDE

    candidates = []
    yolo_crop = _yolo_crop(image_path, img_bgr)
    if yolo_crop is not None:
        candidates.append(("yolo", yolo_crop))
    if is_small:
        candidates.append(("full", img_bgr))

    if not candidates:
        return {"status": "rejected", "reason_code": "NO_NOSE_DETECTED",
                "crop_bgr": None, "validator_score": 0.0,
                "used_yolo": False, "validator_scores": {}}

    best_source = None
    best_crop = None
    best_score = -1.0
    scores = {}
    for src, crop in candidates:
        s = _validator_score(crop)
        scores[src] = s
        if s > best_score:
            best_score = s
            best_source = src
            best_crop = crop

    if best_score < VALIDATOR_THRESHOLD:
        return {"status": "rejected", "reason_code": "VALIDATOR_REJECTED",
                "crop_bgr": None, "validator_score": best_score,
                "used_yolo": (best_source == "yolo"),
                "validator_scores": scores}

    return {"status": "accepted", "reason_code": "OK",
            "crop_bgr": best_crop, "validator_score": best_score,
            "used_yolo": (best_source == "yolo"),
            "validator_scores": scores}