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
VALIDATOR_THRESHOLD  = 0.5
BLUR_THRESHOLD = 15.0  # Minimum sharpness score

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

def _is_blurry(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < BLUR_THRESHOLD, variance

def _enhance_lighting(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def _rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

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
    _load()

    if img_bgr is None:
        if image_path is None:
            return {"status": "rejected", "reason_code": "IMAGE_UNREADABLE", "crop_bgr": None, "validator_score": 0.0}
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return {"status": "rejected", "reason_code": "IMAGE_UNREADABLE", "crop_bgr": None, "validator_score": 0.0}

    # 1. Blur Check
    is_blurry, sharpness = _is_blurry(img_bgr)
    if is_blurry:
        # Try to enhance lighting first to see if it helps
        img_bgr = _enhance_lighting(img_bgr)
        is_blurry, sharpness = _is_blurry(img_bgr)
        if is_blurry:
            return {"status": "rejected", "reason_code": "IMAGE_TOO_BLURRY", "crop_bgr": None, "validator_score": 0.0, "sharpness": sharpness}

    # 2. Candidates: Always check the FULL image (fixes pre-cropped noses)
    candidates = [("full", img_bgr)]
    
    yolo_crop = _yolo_crop(image_path, img_bgr)
    if yolo_crop is not None:
        candidates.append(("yolo", yolo_crop))

    best_source = None
    best_crop = None
    best_score = -1.0
    
    for src, crop in candidates:
        s = _validator_score(crop)
        if s > best_score:
            best_score = s
            best_source = src
            best_crop = crop
            # If the full image is already a perfect nose, accept immediately
            if best_score > 0.90:
                return {"status": "accepted", "reason_code": "OK", "crop_bgr": best_crop, "validator_score": best_score, "used_yolo": (best_source == "yolo")}

    # 3. Rotation Fallback
    if best_score < VALIDATOR_THRESHOLD:
        angles = [-30, -15, 15, 30]
        for angle in angles:
            rotated_img = _rotate_image(img_bgr, angle)
            rotated_crop = _yolo_crop(None, rotated_img)
            if rotated_crop is not None:
                upright_crop = _rotate_image(rotated_crop, -angle)
                s = _validator_score(upright_crop)
                if s > best_score:
                    best_score = s
                    best_crop = upright_crop
                    best_source = f"yolo_rot{angle}"
                    if best_score >= VALIDATOR_THRESHOLD:
                        break

    if best_score < VALIDATOR_THRESHOLD:
        return {"status": "rejected", "reason_code": "VALIDATOR_REJECTED", "crop_bgr": None, "validator_score": best_score}

    return {"status": "accepted", "reason_code": "OK", "crop_bgr": best_crop, "validator_score": best_score, "used_yolo": (best_source == "yolo")}