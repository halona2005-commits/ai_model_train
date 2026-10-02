import os
import cv2
import shutil


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"E:\canine-nose-print-recognition\ai-model"

INPUT_DIR = os.path.join(
    BASE_DIR,
    "random_dog"
)

FEEDBACK_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "face_detection_feedback"
)

IMAGE_DIR = os.path.join(
    FEEDBACK_DIR,
    "images"
)

LABEL_DIR = os.path.join(
    FEEDBACK_DIR,
    "labels"
)

PREVIEW_DIR = os.path.join(
    FEEDBACK_DIR,
    "previews"
)


# ============================================================
# IMAGES THAT NEED FACE CORRECTION
# ============================================================

FEEDBACK_IMAGES = [
    "dog14.jpg",
    "dog15.jpg",
    "dog16.jpg",
    "dog18.jpg",
    "dog19.jpg",
    "dog20.jpg"
]


# ============================================================
# DISPLAY SETTINGS
# ============================================================

# Maximum size of the labeling window.
# The original image is resized only for DISPLAY.
# The original image itself is NOT changed.
MAX_DISPLAY_WIDTH = 1400
MAX_DISPLAY_HEIGHT = 800


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    IMAGE_DIR,
    exist_ok=True
)

os.makedirs(
    LABEL_DIR,
    exist_ok=True
)

os.makedirs(
    PREVIEW_DIR,
    exist_ok=True
)


# ============================================================
# CONVERT PIXEL BOX TO YOLO
# ============================================================

def convert_to_yolo(
    x,
    y,
    w,
    h,
    image_width,
    image_height
):

    center_x = x + (w / 2)
    center_y = y + (h / 2)

    center_x /= image_width
    center_y /= image_height

    w /= image_width
    h /= image_height

    return (
        center_x,
        center_y,
        w,
        h
    )


# ============================================================
# CREATE FULL-IMAGE DISPLAY
# ============================================================

def prepare_display_image(image):

    original_height, original_width = image.shape[:2]

    scale_x = (
        MAX_DISPLAY_WIDTH /
        original_width
    )

    scale_y = (
        MAX_DISPLAY_HEIGHT /
        original_height
    )

    # Use the smaller scale so the ENTIRE image fits.
    scale = min(
        scale_x,
        scale_y,
        1.0
    )

    display_width = int(
        original_width * scale
    )

    display_height = int(
        original_height * scale
    )

    if scale < 1.0:

        display_image = cv2.resize(
            image,
            (
                display_width,
                display_height
            ),
            interpolation=cv2.INTER_AREA
        )

    else:

        display_image = image.copy()

    return display_image, scale


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("              FACE FEEDBACK LABELING")
    print("=" * 70)

    print()
    print("The COMPLETE IMAGE will be fitted inside the window.")
    print()
    print("IMPORTANT:")
    print("1. Draw around the visible DOG FACE.")
    print("2. Include the complete visible muzzle/nose.")
    print("3. Do not cut the nose at the edge.")
    print("4. Include a little surrounding area.")
    print("5. Do NOT draw around the whole dog body.")
    print()
    print("ENTER/SPACE = accept")
    print("C = skip")
    print()

    input("Press ENTER to begin...")

    labeled = 0
    skipped = 0

    # ========================================================
    # PROCESS EACH IMAGE
    # ========================================================

    for index, image_name in enumerate(
        FEEDBACK_IMAGES,
        start=1
    ):

        image_path = os.path.join(
            INPUT_DIR,
            image_name
        )

        print()
        print("=" * 70)
        print(
            f"IMAGE {index}/{len(FEEDBACK_IMAGES)}"
        )
        print(
            f"File: {image_name}"
        )
        print("=" * 70)

        # ----------------------------------------------------
        # Check image
        # ----------------------------------------------------

        if not os.path.exists(image_path):

            print(
                f"[ERROR] Image not found:"
            )

            print(image_path)

            skipped += 1
            continue

        image = cv2.imread(
            image_path
        )

        if image is None:

            print(
                f"[ERROR] Could not read "
                f"{image_name}"
            )

            skipped += 1
            continue

        # Original dimensions
        original_height, original_width = (
            image.shape[:2]
        )

        # ----------------------------------------------------
        # Resize ONLY for display
        # ----------------------------------------------------

        display_image, scale = (
            prepare_display_image(image)
        )

        display_height, display_width = (
            display_image.shape[:2]
        )

        print()
        print(
            f"Original size : "
            f"{original_width} x "
            f"{original_height}"
        )

        print(
            f"Display size  : "
            f"{display_width} x "
            f"{display_height}"
        )

        print(
            f"Display scale : "
            f"{scale:.4f}"
        )

        print()
        print(
            "Draw a box around the COMPLETE "
            "VISIBLE DOG FACE."
        )

        print(
            "Make sure the NOSE and MUZZLE "
            "are inside."
        )

        print()

        # ----------------------------------------------------
        # Create normal resizable window
        # ----------------------------------------------------

        window_name = "Draw DOG FACE"

        cv2.namedWindow(
            window_name,
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            window_name,
            display_width,
            display_height
        )

        # ----------------------------------------------------
        # Draw ROI on resized display image
        # ----------------------------------------------------

        roi = cv2.selectROI(
            window_name,
            display_image,
            showCrosshair=True,
            fromCenter=False
        )

        cv2.destroyAllWindows()

        display_x, display_y, display_w, display_h = [
            int(value)
            for value in roi
        ]

        # ----------------------------------------------------
        # Check skipped
        # ----------------------------------------------------

        if (
            display_w <= 0
            or display_h <= 0
        ):

            print(
                f"[SKIPPED] {image_name}"
            )

            skipped += 1
            continue

        # ====================================================
        # CONVERT DISPLAY COORDINATES
        # BACK TO ORIGINAL IMAGE
        # ====================================================

        x = int(
            display_x / scale
        )

        y = int(
            display_y / scale
        )

        w = int(
            display_w / scale
        )

        h = int(
            display_h / scale
        )

        # Keep coordinates inside original image

        x = max(
            0,
            min(x, original_width - 1)
        )

        y = max(
            0,
            min(y, original_height - 1)
        )

        w = min(
            w,
            original_width - x
        )

        h = min(
            h,
            original_height - y
        )

        if w <= 0 or h <= 0:

            print(
                f"[INVALID BOX] {image_name}"
            )

            skipped += 1
            continue

        # ====================================================
        # CONVERT TO YOLO
        # ====================================================

        (
            center_x,
            center_y,
            yolo_width,
            yolo_height
        ) = convert_to_yolo(
            x,
            y,
            w,
            h,
            original_width,
            original_height
        )

        # ====================================================
        # OUTPUT NAMES
        # ====================================================

        base_name = os.path.splitext(
            image_name
        )[0]

        output_name = (
            f"feedback_{base_name}.jpg"
        )

        label_name = (
            f"feedback_{base_name}.txt"
        )

        output_image = os.path.join(
            IMAGE_DIR,
            output_name
        )

        output_label = os.path.join(
            LABEL_DIR,
            label_name
        )

        preview_path = os.path.join(
            PREVIEW_DIR,
            output_name
        )

        # ====================================================
        # COPY ORIGINAL IMAGE
        # ====================================================

        shutil.copy2(
            image_path,
            output_image
        )

        # ====================================================
        # SAVE YOLO LABEL
        # ====================================================

        with open(
            output_label,
            "w"
        ) as file:

            file.write(
                f"0 "
                f"{center_x:.6f} "
                f"{center_y:.6f} "
                f"{yolo_width:.6f} "
                f"{yolo_height:.6f}\n"
            )

        # ====================================================
        # SAVE PREVIEW ON ORIGINAL IMAGE
        # ====================================================

        preview = image.copy()

        cv2.rectangle(
            preview,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            4
        )

        cv2.putText(
            preview,
            "DOG FACE",
            (x, max(40, y - 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3
        )

        cv2.imwrite(
            preview_path,
            preview
        )

        # ====================================================
        # PRINT RESULT
        # ====================================================

        print()
        print(
            f"[LABELED] {image_name}"
        )

        print(
            f"Original pixel box: "
            f"x={x}, y={y}, "
            f"w={w}, h={h}"
        )

        print(
            f"YOLO box: "
            f"{center_x:.6f} "
            f"{center_y:.6f} "
            f"{yolo_width:.6f} "
            f"{yolo_height:.6f}"
        )

        labeled += 1

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("              LABELING COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Labeled : {labeled}"
    )

    print(
        f"Skipped : {skipped}"
    )

    print()
    print("Images:")
    print(IMAGE_DIR)

    print()
    print("Labels:")
    print(LABEL_DIR)

    print()
    print("Previews:")
    print(PREVIEW_DIR)


if __name__ == "__main__":
    main()