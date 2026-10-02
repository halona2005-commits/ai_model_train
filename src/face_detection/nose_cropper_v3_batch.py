import os
import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"E:\canine-nose-print-recognition\ai-model"

INPUT_DIR = os.path.join(
    BASE_DIR,
    "face_crops"
)

BEST_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "nose_detector",
    "yolo11n_nose_tight",
    "weights",
    "best.pt"
)

LAST_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "nose_detector",
    "yolo11n_nose_augmented",
    "weights",
    "last.pt"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "nose_v3_batch"
)

CROPS_DIR = os.path.join(
    OUTPUT_DIR,
    "crops"
)

VISUALIZED_DIR = os.path.join(
    OUTPUT_DIR,
    "visualized"
)

MASKS_DIR = os.path.join(
    OUTPUT_DIR,
    "masks"
)


# ============================================================
# SETTINGS
# ============================================================

BEST_CONFIDENCE = 0.25
LAST_CONFIDENCE = 0.15

IMAGE_SIZE = 1024

# Small padding around YOLO nose box
CROP_PADDING = 0.10

# Contour threshold
DARK_THRESHOLD = 105

# Only search for contour inside this percentage
# of the YOLO box around its center.
REFINE_RATIO = 0.70

# Contour must occupy a reasonable portion of
# the YOLO box to be accepted.
MIN_CONTOUR_RATIO = 0.12
MAX_CONTOUR_RATIO = 0.95

# Maximum allowed distance of contour center
# from YOLO box center.
MAX_CENTER_SHIFT = 0.35


# ============================================================
# CREATE / CLEAN OUTPUT DIRECTORIES
# ============================================================

for folder in [
    CROPS_DIR,
    VISUALIZED_DIR,
    MASKS_DIR
]:

    os.makedirs(
        folder,
        exist_ok=True
    )


# Remove previous generated results

for folder in [
    CROPS_DIR,
    VISUALIZED_DIR,
    MASKS_DIR
]:

    for filename in os.listdir(folder):

        file_path = os.path.join(
            folder,
            filename
        )

        if os.path.isfile(file_path):

            try:
                os.remove(file_path)
            except:
                pass


# ============================================================
# LOAD MODELS
# ============================================================

print()
print("=" * 70)
print("          NOSE CROPPER V3 - YOLO GUIDED")
print("=" * 70)

print()
print("Loading best.pt...")
best_model = YOLO(BEST_MODEL)

print("Loading last.pt...")
last_model = YOLO(LAST_MODEL)


# ============================================================
# FIND FACE CROPS
# ============================================================

image_files = []

for filename in sorted(
    os.listdir(INPUT_DIR)
):

    if filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):

        image_files.append(filename)


print()
print(
    f"Face crops found: {len(image_files)}"
)

print()
print(
    "Processing nose detections..."
)

print()


successful = 0
failed = 0


# ============================================================
# PROCESS EACH FACE CROP
# ============================================================

for filename in image_files:

    image_path = os.path.join(
        INPUT_DIR,
        filename
    )

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            f"[ERROR] Could not read {filename}"
        )

        failed += 1
        continue


    height, width = image.shape[:2]


    # ========================================================
    # STEP 1
    # BEST NOSE DETECTOR
    # ========================================================

    results = best_model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=BEST_CONFIDENCE,
        verbose=False
    )

    boxes = results[0].boxes


    if boxes is not None and len(boxes) > 0:

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
            boxes.xyxy[
                best_index
            ]
            .cpu()
            .numpy()
        )

        model_used = "best.pt"


    else:

        # ====================================================
        # STEP 2
        # LAST MODEL FALLBACK
        # ====================================================

        results = last_model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            conf=LAST_CONFIDENCE,
            verbose=False
        )

        boxes = results[0].boxes


        if boxes is None or len(boxes) == 0:

            print(
                f"[NO NOSE] {filename}"
            )

            failed += 1
            continue


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
            boxes.xyxy[
                best_index
            ]
            .cpu()
            .numpy()
        )

        model_used = "last.pt"


    # ========================================================
    # YOLO NOSE BOX
    # ========================================================

    x1, y1, x2, y2 = [
        int(v)
        for v in xyxy
    ]


    # Keep inside image

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


    if nose_width <= 0 or nose_height <= 0:

        print(
            f"[INVALID BOX] {filename}"
        )

        failed += 1
        continue


    # ========================================================
    # YOLO BOX CENTER
    # ========================================================

    center_x = (
        x1 + x2
    ) / 2.0

    center_y = (
        y1 + y2
    ) / 2.0


    # ========================================================
    # REFINEMENT REGION
    #
    # We do NOT search the whole face.
    #
    # We search only the central 70% of the YOLO
    # nose box.
    # ========================================================

    refine_width = (
        nose_width * REFINE_RATIO
    )

    refine_height = (
        nose_height * REFINE_RATIO
    )


    refine_x1 = int(
        center_x -
        refine_width / 2
    )

    refine_y1 = int(
        center_y -
        refine_height / 2
    )

    refine_x2 = int(
        center_x +
        refine_width / 2
    )

    refine_y2 = int(
        center_y +
        refine_height / 2
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


    # ========================================================
    # DARK MASK INSIDE REFINEMENT REGION
    # ========================================================

    gray = cv2.cvtColor(
        refine,
        cv2.COLOR_BGR2GRAY
    )


    dark_mask = cv2.inRange(
        gray,
        0,
        DARK_THRESHOLD
    )


    # ========================================================
    # CLEAN MASK
    # ========================================================

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


    # ========================================================
    # FIND CONTOURS
    # ========================================================

    contours, _ = cv2.findContours(
        dark_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    # ========================================================
    # FIND SAFE CONTOUR
    # ========================================================

    best_contour = None
    best_score = -1


    yolo_area = (
        nose_width *
        nose_height
    )


    for contour in contours:

        area = cv2.contourArea(
            contour
        )


        if area <= 0:
            continue


        # ----------------------------------------------------
        # CONTOUR SIZE CHECK
        # ----------------------------------------------------

        contour_ratio = (
            area /
            float(yolo_area)
        )


        if contour_ratio < MIN_CONTOUR_RATIO:
            continue


        if contour_ratio > MAX_CONTOUR_RATIO:
            continue


        # ----------------------------------------------------
        # CONTOUR CENTER
        # ----------------------------------------------------

        moments = cv2.moments(
            contour
        )


        if moments["m00"] == 0:
            continue


        contour_cx = (
            moments["m10"] /
            moments["m00"]
        )

        contour_cy = (
            moments["m01"] /
            moments["m00"]
        )


        # Convert refinement coordinates
        # back to original image coordinates

        contour_cx += refine_x1
        contour_cy += refine_y1


        # ----------------------------------------------------
        # DISTANCE FROM YOLO CENTER
        # ----------------------------------------------------

        dx = abs(
            contour_cx -
            center_x
        ) / max(
            nose_width,
            1
        )

        dy = abs(
            contour_cy -
            center_y
        ) / max(
            nose_height,
            1
        )


        if dx > MAX_CENTER_SHIFT:
            continue

        if dy > MAX_CENTER_SHIFT:
            continue


        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        center_score = 1.0 - (
            dx + dy
        ) / 2.0


        size_score = min(
            contour_ratio,
            1.0
        )


        score = (
            center_score *
            0.7
            +
            size_score *
            0.3
        )


        if score > best_score:

            best_score = score
            best_contour = contour


    # ========================================================
    # FINAL CROP BOX
    #
    # IMPORTANT:
    #
    # If contour is trustworthy:
    #     use contour
    #
    # Otherwise:
    #     use YOLO box
    #
    # The contour can NEVER escape the YOLO box.
    # ========================================================

    if best_contour is not None:

        contour_x, contour_y, contour_w, contour_h = (
            cv2.boundingRect(
                best_contour
            )
        )


        contour_x += refine_x1
        contour_y += refine_y1


        # Clamp contour to YOLO box

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
            contour_x +
            contour_w
        )

        final_y2 = min(
            y2,
            contour_y +
            contour_h
        )


        final_w = (
            final_x2 -
            final_x1
        )

        final_h = (
            final_y2 -
            final_y1
        )


        # Safety check

        if (
            final_w <= 0
            or
            final_h <= 0
        ):

            final_x1 = x1
            final_y1 = y1
            final_x2 = x2
            final_y2 = y2

            crop_source = "YOLO"


        else:

            crop_source = "REFINED"


    else:

        final_x1 = x1
        final_y1 = y1
        final_x2 = x2
        final_y2 = y2

        crop_source = "YOLO"


    # ========================================================
    # ADD SMALL PADDING
    # ========================================================

    crop_width = (
        final_x2 -
        final_x1
    )

    crop_height = (
        final_y2 -
        final_y1
    )


    pad_x = int(
        crop_width *
        CROP_PADDING
    )

    pad_y = int(
        crop_height *
        CROP_PADDING
    )


    final_x1 -= pad_x
    final_y1 -= pad_y
    final_x2 += pad_x
    final_y2 += pad_y


    # ========================================================
    # MAKE CONTROLLED SQUARE CROP
    # ========================================================

    crop_width = final_x2 - final_x1
    crop_height = final_y2 - final_y1

    # Use a slightly larger dimension, but do not
    # excessively expand the crop.
    side = int(
        max(crop_width, crop_height) * 1.02
    )

    crop_center_x = (
        final_x1 + final_x2
    ) / 2.0

    crop_center_y = (
        final_y1 + final_y2
    ) / 2.0

    final_x1 = int(
        crop_center_x - side / 2
    )

    final_y1 = int(
        crop_center_y - side / 2
    )

    final_x2 = int(
        crop_center_x + side / 2
    )

    final_y2 = int(
        crop_center_y + side / 2
    )

    # ========================================================
    # IMAGE BOUNDARY
    # ========================================================

    if final_x1 < 0:

        final_x2 += -final_x1
        final_x1 = 0


    if final_y1 < 0:

        final_y2 += -final_y1
        final_y1 = 0


    if final_x2 > width:

        final_x1 -= (
            final_x2 -
            width
        )

        final_x2 = width


    if final_y2 > height:

        final_y1 -= (
            final_y2 -
            height
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


    # ========================================================
    # FINAL NOSE CROP
    # ========================================================

    nose_crop = image[
        final_y1:final_y2,
        final_x1:final_x2
    ]


    if nose_crop.size == 0:

        print(
            f"[EMPTY CROP] {filename}"
        )

        failed += 1
        continue


    # ========================================================
    # VISUALIZATION
    # ========================================================

    visualized = image.copy()


    # Blue = YOLO box

    cv2.rectangle(
        visualized,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        2
    )


    # Yellow = refinement region

    cv2.rectangle(
        visualized,
        (refine_x1, refine_y1),
        (refine_x2, refine_y2),
        (0, 255, 255),
        2
    )


    # Red = final crop

    cv2.rectangle(
        visualized,
        (final_x1, final_y1),
        (final_x2, final_y2),
        (0, 0, 255),
        2
    )


    # ========================================================
    # SAVE
    # ========================================================

    base_name = os.path.splitext(
        filename
    )[0]


    crop_path = os.path.join(
        CROPS_DIR,
        base_name +
        "_nose.jpg"
    )


    visualization_path = os.path.join(
        VISUALIZED_DIR,
        base_name +
        "_box.jpg"
    )


    mask_path = os.path.join(
        MASKS_DIR,
        base_name +
        "_mask.jpg"
    )


    cv2.imwrite(
        crop_path,
        nose_crop
    )


    cv2.imwrite(
        visualization_path,
        visualized
    )


    cv2.imwrite(
        mask_path,
        dark_mask
    )


    print(
        f"[{model_used}] "
        f"{filename} "
        f"-> confidence={confidence:.2f} "
        f"-> crop={crop_source}"
    )


    successful += 1


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("       NOSE CROPPER V3 - YOLO GUIDED COMPLETE")
print("=" * 70)

print()
print(
    f"Successful nose crops : {successful}"
)

print(
    f"Failed                : {failed}"
)

print()
print("Crops:")
print(CROPS_DIR)

print()
print("Visualizations:")
print(VISUALIZED_DIR)

print()
print("Masks:")
print(MASKS_DIR)

print()
print("=" * 70)