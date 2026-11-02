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
BLUR_THRESHOLD = 5.0  # Lowered from 15.0 to accept slightly shaky images

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

def _enhance_image(img_bgr):
    """Fixes low lighting and applies mild sharpening for shaky images."""
    # 1. Gamma Correction (Brightens dark images)
    gamma = 1.5
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    brightened = cv2.LUT(img_bgr, table)
    
    # 2. CLAHE (Local contrast enhancement)
    lab = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    
    # 3. Unsharp Mask (Slight sharpening for shaky images)
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
    
    return sharpened

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

    # 1. ENHANCE LIGHTING & SHARPNESS (Runs on every image)
    enhanced_img = _enhance_image(img_bgr)

    # 2. BLUR CHECK (Uses the lower threshold of 5.0)
    is_blurry, sharpness = _is_blurry(enhanced_img)
    if is_blurry:
        return {"status": "rejected", "reason_code": "IMAGE_TOO_BLURRY", "crop_bgr": None, "validator_score": 0.0, "sharpness": sharpness}

    # 3. CHECK IF ALREADY A CROPPED NOSE (Full Image First)
    # If the user uploads a pre-cropped nose, YOLO might fail, so we check the full image first.
    s_full = _validator_score(enhanced_img)
    if s_full >= VALIDATOR_THRESHOLD:
        return {"status": "accepted", "reason_code": "OK", "crop_bgr": enhanced_img, "validator_score": s_full, "used_yolo": False}

    # 4. TRY YOLO CROP (Full Dog Image)
    yolo_crop = _yolo_crop(image_path, enhanced_img)
    if yolo_crop is not None:
        s_crop = _validator_score(yolo_crop)
        if s_crop >= VALIDATOR_THRESHOLD:
            return {"status": "accepted", "reason_code": "OK", "crop_bgr": yolo_crop, "validator_score": s_crop, "used_yolo": True}

    # 5. ROTATION FALLBACK (Up to 90°)
    # If the dog is heavily tilted, we rotate the image, find the nose, and un-rotate the crop.
    angles = [-90, -75, -60, -45, -30, -15, 15, 30, 45, 60, 75, 90]
    for angle in angles:
        rotated_img = _rotate_image(enhanced_img, angle)
        
        # Check if the rotated FULL image is a nose (for pre-cropped tilted noses)
        s_rot_full = _validator_score(rotated_img)
        if s_rot_full >= VALIDATOR_THRESHOLD:
            # Un-rotate it back to upright
            upright_crop = _rotate_image(rotated_img, -angle)
            return {"status": "accepted", "reason_code": "OK", "crop_bgr": upright_crop, "validator_score": s_rot_full, "used_yolo": False}
            
        # Check YOLO crop on the rotated image
        rot_crop = _yolo_crop(None, rotated_img)
        if rot_crop is not None:
            upright_crop = _rotate_image(rot_crop, -angle)
            s_rot_crop = _validator_score(upright_crop)
            if s_rot_crop >= VALIDATOR_THRESHOLD:
                return {"status": "accepted", "reason_code": "OK", "crop_bgr": upright_crop, "validator_score": s_rot_crop, "used_yolo": True}

    # 6. FINAL REJECTION
    return {"status": "rejected", "reason_code": "VALIDATOR_REJECTED", "crop_bgr": None, "validator_score": max(s_full, s_crop if yolo_crop is not None else 0)}