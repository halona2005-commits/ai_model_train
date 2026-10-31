"""
Single validated nose-preparation function used by BOTH
register.py and recognize.py.
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

def _rotate_image(image, angle):
    """Rotates an image around its center."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def _get_crop(img, image_path=None):
    """Runs YOLO on the image and returns the crop."""
    # If image_path is provided, use it (YOLO is faster with paths)
    yolo_input = str(image_path) if image_path is not None else img
    results = _yolo(yolo_input, conf=0.25, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    i = int(np.argmax(confs))
    x1, y1, x2, y2 = xyxy[i]
    h, w = img.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if (x2 - x1) < 16 or (y2 - y1) < 16:
        return None
    return img[y1:y2, x1:x2]

def prepare_nose(image_path=None, img_bgr=None):
    _load()

    if img_bgr is None:
        if image_path is None:
            return {"status": "rejected", "reason_code": "IMAGE_UNREADABLE", "crop_bgr": None, "validator_score": 0.0}
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return {"status": "rejected", "reason_code": "IMAGE_UNREADABLE", "crop_bgr": None, "validator_score": 0.0}

    # 1. Try original image first (0 degrees)
    original_crop = _get_crop(img_bgr, image_path)
    if original_crop is not None:
        s = _validator_score(original_crop)
        if s >= VALIDATOR_THRESHOLD:
            return {"status": "accepted", "reason_code": "OK", "crop_bgr": original_crop, "validator_score": s, "used_yolo": True}

    # 2. If YOLO fails or Validator rejects, try rotating the FULL image
    # This fixes both YOLO failure (positive rot) and Embedder failure (negative rot)
    angles = [-30, -20, -15, 15, 20, 30]
    for angle in angles:
        rotated_img = _rotate_image(img_bgr, angle)
        
        # Run YOLO on the rotated image
        rotated_crop = _get_crop(rotated_img, None)
        if rotated_crop is not None:
            # Un-rotate the CROP so the Embedder sees an upright nose!
            upright_crop = _rotate_image(rotated_crop, -angle)
            
            s = _validator_score(upright_crop)
            if s >= VALIDATOR_THRESHOLD:
                return {
                    "status": "accepted", 
                    "reason_code": "OK", 
                    "crop_bgr": upright_crop, 
                    "validator_score": s, 
                    "used_yolo": True
                }

    # 3. If everything fails, reject
    return {"status": "rejected", "reason_code": "NO_NOSE_DETECTED", "crop_bgr": None, "validator_score": 0.0}