import sys
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model


# ============================================================
# BASE PATH
# ============================================================

BASE_DIR = Path(
    r"E:\canine-nose-print-recognition\ai-model"
)


# ============================================================
# MODEL PATHS
# ============================================================

FACE_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "face_detector"
    / "yolo11n_face_final"
    / "weights"
    / "best.pt"
)

BEST_NOSE_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "nose_detector"
    / "yolo11n_nose_tight"
    / "weights"
    / "best.pt"
)

LAST_NOSE_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "nose_detector"
    / "yolo11n_nose_augmented"
    / "weights"
    / "last.pt"
)

NOSE_VALIDATOR_PATH = (
    BASE_DIR
    / "models"
    / "dog_nose_validator.keras"
)

REALESRGAN_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "RealESRGAN_x4plus.pth"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_ROOT = (
    BASE_DIR
    / "pipeline_results"
)


# ============================================================
# SETTINGS
# ============================================================

FACE_CONFIDENCE = 0.25

BEST_NOSE_CONFIDENCE = 0.25
LAST_NOSE_CONFIDENCE = 0.15

FACE_IMAGE_SIZE = 640
NOSE_IMAGE_SIZE = 1024

FACE_PADDING = 0.15

CROP_PADDING = 0.10
REFINE_RATIO = 0.70

DARK_THRESHOLD = 105

MIN_CONTOUR_RATIO = 0.12
MAX_CONTOUR_RATIO = 0.95

MAX_CENTER_SHIFT = 0.35


# ============================================================
# IMAGE QUALITY
# ============================================================

QUALITY_MIN_WIDTH = 224
QUALITY_MIN_HEIGHT = 224

# Prototype threshold.
# This can be calibrated later using an evaluation dataset.
QUALITY_MIN_LAPLACIAN = 50.0


# ============================================================
# DOG NOSE VALIDATOR
# ============================================================

# The validator was trained with:
#
# 0 = dog_nose_present
# 1 = not_dog_nose
#
# Therefore:
#
# prediction < 0.50 -> DOG_NOSE_PRESENT
# prediction >= 0.50 -> NOT_DOG_NOSE

NOSE_VALIDATOR_THRESHOLD = 0.50


# ============================================================
# LOAD MODELS
# ============================================================

print()
print("=" * 70)
print("              CANINE NOSE PRINT AI PIPELINE")
print("=" * 70)

print()
print("Loading face detector...")
face_model = YOLO(
    str(FACE_MODEL_PATH)
)

print("Loading best nose detector...")
best_nose_model = YOLO(
    str(BEST_NOSE_MODEL_PATH)
)

print("Loading last nose detector...")
last_nose_model = YOLO(
    str(LAST_NOSE_MODEL_PATH)
)

print("Loading dog-nose validator...")

nose_validator = load_model(
    str(NOSE_VALIDATOR_PATH),
    custom_objects={
        "preprocess_input": preprocess_input
    },
    compile=False
)

print("Loading Real-ESRGAN...")

rrdb_model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4
)

upsampler = RealESRGANer(
    scale=4,
    model_path=str(REALESRGAN_MODEL_PATH),
    model=rrdb_model,

    # Stable setting used successfully
    # on all 19 runtime images.
    tile=128,

    tile_pad=10,
    pre_pad=0,
    half=False
)

print("Loading ResNet50...")

resnet_model = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

print("All models loaded.")


# ============================================================
# STEP 1
# IMAGE QUALITY CHECK
# ============================================================

def check_image_quality(image):

    if image is None or image.size == 0:
        return (
            False,
            "IMAGE_NOT_READABLE",
            0.0
        )

    height, width = image.shape[:2]

    if (
        width < QUALITY_MIN_WIDTH
        or height < QUALITY_MIN_HEIGHT
    ):
        return (
            False,
            f"IMAGE_TOO_SMALL ({width}x{height})",
            0.0
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    laplacian_variance = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    if laplacian_variance < QUALITY_MIN_LAPLACIAN:
        return (
            False,
            "LOW_IMAGE_QUALITY / TOO_BLURRY",
            laplacian_variance
        )

    return (
        True,
        "QUALITY_PASS",
        laplacian_variance
    )


# ============================================================
# STEP 2A
# DOG FACE DETECTION
#
# Used only as part of Step 2 validation/routing.
#
# If a dog face is found:
#     classify input as FULL_DOG
#
# If no dog face is found:
#     test the image with the trained dog-nose validator.
# ============================================================

def detect_dog_face(image):

    height, width = image.shape[:2]

    results = face_model.predict(
        source=image,
        imgsz=FACE_IMAGE_SIZE,
        conf=FACE_CONFIDENCE,
        verbose=False
    )

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return None, None

    confidences = (
        boxes.conf
        .cpu()
        .numpy()
    )

    best_index = int(
        np.argmax(confidences)
    )

    confidence = float(
        confidences[best_index]
    )

    xyxy = (
        boxes.xyxy[best_index]
        .cpu()
        .numpy()
    )

    x1, y1, x2, y2 = [
        int(v)
        for v in xyxy
    ]

    x1 = max(
        0,
        min(x1, width - 1)
    )

    y1 = max(
        0,
        min(y1, height - 1)
    )

    x2 = max(
        0,
        min(x2, width)
    )

    y2 = max(
        0,
        min(y2, height)
    )

    if x2 <= x1 or y2 <= y1:
        return None, None

    return (
        np.array(
            [x1, y1, x2, y2],
            dtype=np.int32
        ),
        confidence
    )


# ============================================================
# STEP 2B
# DOG-NOSE VALIDATION
#
# IMPORTANT:
# This model was trained on nose images.
#
# It is used ONLY when the face detector does not find
# a full dog face.
#
# It determines whether the uploaded image itself
# is a dog-nose image.
# ============================================================

def validate_uploaded_nose(image):

    image_224 = cv2.resize(
        image,
        (224, 224),
        interpolation=cv2.INTER_LANCZOS4
    )

    image_224 = cv2.cvtColor(
        image_224,
        cv2.COLOR_BGR2RGB
    )

    image_224 = image_224.astype(
        np.float32
    )

    # EXACT preprocessing used during validator training.
    image_224 = preprocess_input(
        image_224
    )

    image_224 = np.expand_dims(
        image_224,
        axis=0
    )

    prediction = float(
        nose_validator.predict(
            image_224,
            verbose=0
        )[0][0]
    )

    if prediction >= NOSE_VALIDATOR_THRESHOLD:

        return (
            False,
            "NOT_DOG_NOSE",
            prediction
        )

    confidence = 1.0 - prediction

    return (
        True,
        "DOG_NOSE_PRESENT",
        confidence
    )


# ============================================================
# STEP 2
# DOG VALIDATION / INPUT ROUTING
# ============================================================

def validate_and_route_input(image):

    # --------------------------------------------------------
    # First check whether this is a full dog image.
    # --------------------------------------------------------

    face_box, face_confidence = (
        detect_dog_face(image)
    )

    if face_box is not None:

        return {
            "input_type": "FULL_DOG",
            "face_box": face_box,
            "face_confidence": face_confidence,
            "nose_validation": None,
            "nose_validation_confidence": None
        }

    # --------------------------------------------------------
    # No dog face found.
    #
    # Now determine whether the uploaded image itself
    # is a dog nose.
    # --------------------------------------------------------

    (
        is_dog_nose,
        validation_result,
        validation_confidence
    ) = validate_uploaded_nose(image)

    if is_dog_nose:

        return {
            "input_type": "DOG_NOSE",
            "face_box": None,
            "face_confidence": None,
            "nose_validation": validation_result,
            "nose_validation_confidence": validation_confidence
        }

    return {
        "input_type": "INVALID",
        "face_box": None,
        "face_confidence": None,
        "nose_validation": validation_result,
        "nose_validation_confidence": validation_confidence
    }


# ============================================================
# STEP 3
# CROP DOG FACE
# ============================================================

def crop_face(image, face_box):

    height, width = image.shape[:2]

    x1, y1, x2, y2 = [
        int(v)
        for v in face_box
    ]

    face_width = x2 - x1
    face_height = y2 - y1

    if face_width <= 0 or face_height <= 0:
        return None

    pad_x = int(
        face_width * FACE_PADDING
    )

    pad_y = int(
        face_height * FACE_PADDING
    )

    x1 -= pad_x
    y1 -= pad_y
    x2 += pad_x
    y2 += pad_y

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(width, x2)
    y2 = min(height, y2)

    face_crop = image[
        y1:y2,
        x1:x2
    ]

    if face_crop.size == 0:
        return None

    return face_crop


# ============================================================
# STEP 4
# NOSE DETECTION
#
# best.pt first
# last.pt only if best.pt detects nothing
# ============================================================

def detect_nose(face_image):

    # --------------------------------------------------------
    # FIRST: best.pt
    # --------------------------------------------------------

    results = best_nose_model.predict(
        source=face_image,
        imgsz=NOSE_IMAGE_SIZE,
        conf=BEST_NOSE_CONFIDENCE,
        verbose=False
    )

    boxes = results[0].boxes

    if (
        boxes is not None
        and len(boxes) > 0
    ):

        confidences = (
            boxes.conf
            .cpu()
            .numpy()
        )

        best_index = int(
            np.argmax(confidences)
        )

        confidence = float(
            confidences[best_index]
        )

        xyxy = (
            boxes.xyxy[best_index]
            .cpu()
            .numpy()
        )

        return (
            xyxy,
            confidence,
            "best.pt"
        )

    # --------------------------------------------------------
    # SECOND: last.pt FALLBACK
    # --------------------------------------------------------

    results = last_nose_model.predict(
        source=face_image,
        imgsz=NOSE_IMAGE_SIZE,
        conf=LAST_NOSE_CONFIDENCE,
        verbose=False
    )

    boxes = results[0].boxes

    if (
        boxes is None
        or len(boxes) == 0
    ):
        return (
            None,
            None,
            None
        )

    confidences = (
        boxes.conf
        .cpu()
        .numpy()
    )

    best_index = int(
        np.argmax(confidences)
    )

    confidence = float(
        confidences[best_index]
    )

    xyxy = (
        boxes.xyxy[best_index]
        .cpu()
        .numpy()
    )

    return (
        xyxy,
        confidence,
        "last.pt"
    )


# ============================================================
# STEP 5
# NOSE CROPPER V3
# ============================================================

def crop_nose_v3(
    image,
    nose_box
):

    height, width = image.shape[:2]

    x1, y1, x2, y2 = [
        int(v)
        for v in nose_box
    ]

    x1 = max(
        0,
        min(x1, width - 1)
    )

    y1 = max(
        0,
        min(y1, height - 1)
    )

    x2 = max(
        0,
        min(x2, width)
    )

    y2 = max(
        0,
        min(y2, height)
    )

    nose_width = x2 - x1
    nose_height = y2 - y1

    if (
        nose_width <= 0
        or nose_height <= 0
    ):
        return None

    center_x = (
        x1 + x2
    ) / 2.0

    center_y = (
        y1 + y2
    ) / 2.0

    refine_width = (
        nose_width
        * REFINE_RATIO
    )

    refine_height = (
        nose_height
        * REFINE_RATIO
    )

    refine_x1 = int(
        center_x
        - refine_width / 2
    )

    refine_y1 = int(
        center_y
        - refine_height / 2
    )

    refine_x2 = int(
        center_x
        + refine_width / 2
    )

    refine_y2 = int(
        center_y
        + refine_height / 2
    )

    refine_x1 = max(
        x1,
        refine_x1
    )

    refine_y1 = max(
        y1,
        refine_y1
    )

    refine_x2 = min(
        x2,
        refine_x2
    )

    refine_y2 = min(
        y2,
        refine_y2
    )

    refine = image[
        refine_y1:refine_y2,
        refine_x1:refine_x2
    ]

    if refine.size == 0:
        return None

    gray = cv2.cvtColor(
        refine,
        cv2.COLOR_BGR2GRAY
    )

    dark_mask = cv2.inRange(
        gray,
        0,
        DARK_THRESHOLD
    )

    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    large_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_OPEN,
        small_kernel,
        iterations=1
    )

    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_CLOSE,
        large_kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        dark_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best_contour = None
    best_score = -1

    yolo_area = (
        nose_width
        * nose_height
    )

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area <= 0:
            continue

        contour_ratio = (
            area
            / float(yolo_area)
        )

        if contour_ratio < MIN_CONTOUR_RATIO:
            continue

        if contour_ratio > MAX_CONTOUR_RATIO:
            continue

        moments = cv2.moments(
            contour
        )

        if moments["m00"] == 0:
            continue

        contour_cx = (
            moments["m10"]
            / moments["m00"]
        )

        contour_cy = (
            moments["m01"]
            / moments["m00"]
        )

        contour_cx += refine_x1
        contour_cy += refine_y1

        dx = (
            abs(
                contour_cx
                - center_x
            )
            / max(nose_width, 1)
        )

        dy = (
            abs(
                contour_cy
                - center_y
            )
            / max(nose_height, 1)
        )

        if dx > MAX_CENTER_SHIFT:
            continue

        if dy > MAX_CENTER_SHIFT:
            continue

        center_score = (
            1.0
            - (dx + dy) / 2.0
        )

        size_score = min(
            contour_ratio,
            1.0
        )

        score = (
            center_score * 0.7
            + size_score * 0.3
        )

        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is not None:

        (
            contour_x,
            contour_y,
            contour_w,
            contour_h
        ) = cv2.boundingRect(
            best_contour
        )

        contour_x += refine_x1
        contour_y += refine_y1

        final_x1 = max(
            x1,
            contour_x
        )

        final_y1 = max(
            y1,
            contour_y
        )

        final_x2 = min(
            x2,
            contour_x + contour_w
        )

        final_y2 = min(
            y2,
            contour_y + contour_h
        )

        final_w = (
            final_x2
            - final_x1
        )

        final_h = (
            final_y2
            - final_y1
        )

        if (
            final_w <= 0
            or final_h <= 0
        ):
            final_x1 = x1
            final_y1 = y1
            final_x2 = x2
            final_y2 = y2

    else:

        final_x1 = x1
        final_y1 = y1
        final_x2 = x2
        final_y2 = y2

    crop_width = (
        final_x2
        - final_x1
    )

    crop_height = (
        final_y2
        - final_y1
    )

    pad_x = int(
        crop_width
        * CROP_PADDING
    )

    pad_y = int(
        crop_height
        * CROP_PADDING
    )

    final_x1 -= pad_x
    final_y1 -= pad_y
    final_x2 += pad_x
    final_y2 += pad_y

    crop_width = (
        final_x2
        - final_x1
    )

    crop_height = (
        final_y2
        - final_y1
    )

    side = int(
        max(
            crop_width,
            crop_height
        ) * 1.02
    )

    crop_center_x = (
        final_x1
        + final_x2
    ) / 2.0

    crop_center_y = (
        final_y1
        + final_y2
    ) / 2.0

    final_x1 = int(
        crop_center_x
        - side / 2
    )

    final_y1 = int(
        crop_center_y
        - side / 2
    )

    final_x2 = int(
        crop_center_x
        + side / 2
    )

    final_y2 = int(
        crop_center_y
        + side / 2
    )

    if final_x1 < 0:
        final_x2 += -final_x1
        final_x1 = 0

    if final_y1 < 0:
        final_y2 += -final_y1
        final_y1 = 0

    if final_x2 > width:
        final_x1 -= (
            final_x2 - width
        )
        final_x2 = width

    if final_y2 > height:
        final_y1 -= (
            final_y2 - height
        )
        final_y2 = height

    final_x1 = max(
        0,
        final_x1
    )

    final_y1 = max(
        0,
        final_y1
    )

    final_x2 = min(
        width,
        final_x2
    )

    final_y2 = min(
        height,
        final_y2
    )

    nose_crop = image[
        final_y1:final_y2,
        final_x1:final_x2
    ]

    if nose_crop.size == 0:
        return None

    return nose_crop


# ============================================================
# STEP 6
# NOSE ENHANCEMENT
# ============================================================

def deblur_image(
    image,
    iterations=10
):

    image_float = (
        image.astype(
            np.float32
        ) / 255.0
    )

    size = 5
    sigma = 0.8

    x = np.arange(
        -(size // 2),
        size // 2 + 1
    )

    xx, yy = np.meshgrid(
        x,
        x
    )

    kernel = np.exp(
        -(xx ** 2 + yy ** 2)
        / (2 * sigma ** 2)
    )

    kernel /= kernel.sum()

    kernel = kernel.astype(
        np.float32
    )

    flipped_kernel = cv2.flip(
        kernel,
        -1
    )

    result = image_float.copy()

    for _ in range(iterations):

        estimated = cv2.filter2D(
            result,
            -1,
            kernel
        )

        estimated = np.maximum(
            estimated,
            1e-6
        )

        relative = (
            image_float
            / estimated
        )

        correction = cv2.filter2D(
            relative,
            -1,
            flipped_kernel
        )

        result *= correction

        result = np.clip(
            result,
            0,
            1
        )

    return (
        result * 255
    ).astype(
        np.uint8
    )


def enhance_nose(image):

    enhanced, _ = (
        upsampler.enhance(
            image,
            outscale=4
        )
    )

    gray = cv2.cvtColor(
        enhanced,
        cv2.COLOR_BGR2GRAY
    )

    brightness = np.mean(gray)

    if brightness < 100:

        gamma = 0.75

        table = np.array([
            (
                (i / 255.0) ** gamma
            ) * 255
            for i in range(256)
        ]).astype(
            np.uint8
        )

        enhanced = cv2.LUT(
            enhanced,
            table
        )

    elif brightness > 180:

        gamma = 1.20

        table = np.array([
            (
                (i / 255.0) ** gamma
            ) * 255
            for i in range(256)
        ]).astype(
            np.uint8
        )

        enhanced = cv2.LUT(
            enhanced,
            table
        )

    enhanced = cv2.resize(
        enhanced,
        (224, 224),
        interpolation=cv2.INTER_LANCZOS4
    )

    enhanced = deblur_image(
        enhanced,
        iterations=10
    )

    blurred = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        0.7
    )

    enhanced = cv2.addWeighted(
        enhanced,
        1.7,
        blurred,
        -0.7,
        0
    )

    enhanced = np.clip(
        enhanced,
        0,
        255
    ).astype(
        np.uint8
    )

    return enhanced


# ============================================================
# STEP 7
# RESNET50 FEATURE EXTRACTION
# ============================================================

def extract_features(image):

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (224, 224)
    )

    image = image.astype(
        np.float32
    )

    # EXACT ImageNet ResNet50 preprocessing.
    image = preprocess_input(
        image
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    features = (
        resnet_model.predict(
            image,
            verbose=0
        )
    )

    if features.shape != (
        1,
        2048
    ):
        raise ValueError(
            "Unexpected feature shape: "
            f"{features.shape}"
        )

    return features


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(input_path):

    input_path = Path(
        input_path
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input image not found: "
            f"{input_path}"
        )

    image_name = input_path.stem

    output_dir = (
        OUTPUT_ROOT
        / image_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image = cv2.imread(
        str(input_path)
    )

    if image is None:
        raise ValueError(
            f"Could not read image: "
            f"{input_path}"
        )

    print()
    print("=" * 70)
    print(
        f"INPUT: {input_path.name}"
    )
    print("=" * 70)

    # ========================================================
    # STEP 1
    # IMAGE QUALITY
    # ========================================================

    print()
    print(
        "[1/6] Checking image quality..."
    )

    (
        quality_ok,
        quality_status,
        quality_score
    ) = check_image_quality(
        image
    )

    print(
        f"Laplacian variance: "
        f"{quality_score:.2f}"
    )

    if not quality_ok:

        print(
            f"RESULT: {quality_status}"
        )

        return False

    print(
        "Image quality: PASS"
    )

    # ========================================================
    # STEP 2
    # DOG VALIDATION + INPUT ROUTING
    # ========================================================

    print()
    print(
        "[2/6] Validating input..."
    )

    route = validate_and_route_input(
        image
    )

    input_type = route["input_type"]

    if input_type == "INVALID":

        print(
            f"Validation result: "
            f"{route['nose_validation']} "
            f"(confidence="
            f"{route['nose_validation_confidence']:.4f})"
        )

        print(
            "RESULT: INVALID_INPUT"
        )

        return False

    if input_type == "FULL_DOG":

        print(
            "Validation result: FULL_DOG"
        )

        print(
            f"Dog face confidence: "
            f"{route['face_confidence']:.2f}"
        )

    elif input_type == "DOG_NOSE":

        print(
            "Validation result: DOG_NOSE"
        )

        print(
            f"Nose validation confidence: "
            f"{route['nose_validation_confidence']:.4f}"
        )

    # ========================================================
    # FULL DOG PATH
    # ========================================================

    if input_type == "FULL_DOG":

        # ----------------------------------------------------
        # STEP 3
        # CROP DOG FACE
        # ----------------------------------------------------

        print()
        print(
            "[3/6] Cropping dog face..."
        )

        face_crop = crop_face(
            image,
            route["face_box"]
        )

        if face_crop is None:

            print(
                "RESULT: FACE_CROP_FAILED"
            )

            return False

        face_path = (
            output_dir
            / "face_crop.jpg"
        )

        cv2.imwrite(
            str(face_path),
            face_crop
        )

        print(
            f"Face crop saved: "
            f"{face_crop.shape[1]} x "
            f"{face_crop.shape[0]}"
        )

        # ----------------------------------------------------
        # STEP 4
        # NOSE DETECTION + CROP
        # ----------------------------------------------------

        print()
        print(
            "[4/6] Detecting and "
            "cropping nose..."
        )

        (
            nose_box,
            nose_confidence,
            nose_model
        ) = detect_nose(
            face_crop
        )

        if nose_box is None:

            print(
                "RESULT: "
                "DOG_FACE_DETECTED_BUT_"
                "NOSE_NOT_FOUND"
            )

            return False

        print(
            f"Nose detected using "
            f"{nose_model} "
            f"(confidence="
            f"{nose_confidence:.2f})"
        )

        nose_crop = crop_nose_v3(
            face_crop,
            nose_box
        )

        if nose_crop is None:

            print(
                "RESULT: NOSE_CROP_FAILED"
            )

            return False

        nose_crop_path = (
            output_dir
            / "nose_crop.jpg"
        )

        cv2.imwrite(
            str(nose_crop_path),
            nose_crop
        )

        print(
            f"Nose crop saved: "
            f"{nose_crop.shape[1]} x "
            f"{nose_crop.shape[0]}"
        )

    # ========================================================
    # DIRECT DOG NOSE PATH
    # ========================================================

    else:

        # ----------------------------------------------------
        # Already a dog-nose image.
        #
        # DO NOT crop.
        # DO NOT run nose detector.
        # ----------------------------------------------------

        print()
        print(
            "[3/6] Direct dog-nose input..."
        )

        nose_crop = image.copy()

        nose_crop_path = (
            output_dir
            / "nose_input.jpg"
        )

        cv2.imwrite(
            str(nose_crop_path),
            nose_crop
        )

        print(
            "Nose cropping skipped."
        )

        print(
            "Using uploaded image directly."
        )

    # ========================================================
    # STEP 5
    # ENHANCEMENT
    # ========================================================

    print()
    print(
        "[5/6] Enhancing nose..."
    )

    enhanced_nose = enhance_nose(
        nose_crop
    )

    enhanced_path = (
        output_dir
        / "nose_enhanced.png"
    )

    cv2.imwrite(
        str(enhanced_path),
        enhanced_nose
    )

    print(
        "Enhanced image: 224 x 224"
    )

    # ========================================================
    # STEP 6
    # RESNET50 + SAVE FEATURE
    # ========================================================

    print()
    print(
        "[6/6] Extracting "
        "ResNet50 features..."
    )

    features = extract_features(
        enhanced_nose
    )

    print(
        f"Feature vector shape: "
        f"{features.shape}"
    )

    feature_path = (
        output_dir
        / "feature_2048.npy"
    )

    np.save(
        feature_path,
        features[0]
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print("=" * 70)
    print(
        "                 PIPELINE COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        f"Input image       : "
        f"{input_path}"
    )

    print(
        f"Quality score     : "
        f"{quality_score:.2f}"
    )

    print(
        f"Input type        : "
        f"{input_type}"
    )

    if input_type == "FULL_DOG":

        print(
            f"Face confidence   : "
            f"{route['face_confidence']:.2f}"
        )

        print(
            f"Nose detector     : "
            f"{nose_model}"
        )

        print(
            f"Nose confidence   : "
            f"{nose_confidence:.2f}"
        )

    else:

        print(
            f"Nose validation   : "
            f"{route['nose_validation']}"
        )

        print(
            f"Nose confidence   : "
            f"{route['nose_validation_confidence']:.4f}"
        )

    print(
        f"Feature shape     : "
        f"{features.shape}"
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        output_dir
    )

    print()
    print(
        "2048-D FEATURE VECTOR READY"
    )

    print("=" * 70)

    return True


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print()
        print("Usage:")
        print(
            r'python src\pipeline\run_pipeline.py "path\to\image.jpg"'
        )

        sys.exit(1)

    input_image = sys.argv[1]

    success = run_pipeline(
        input_image
    )

    if not success:
        sys.exit(1)