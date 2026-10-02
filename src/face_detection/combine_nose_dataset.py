import os
import cv2
import shutil
import random
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    r"E:\canine-nose-print-recognition\ai-model"
)

# Your 41 new angled images
EXTRA_DATASET = (
    BASE_DIR
    / "dataset"
    / "nose_detection_extra"
)

EXTRA_IMAGES = (
    EXTRA_DATASET
    / "images"
)

EXTRA_LABELS = (
    EXTRA_DATASET
    / "labels"
)

EXTRA_PREVIEWS = (
    EXTRA_DATASET
    / "previews"
)

# Existing nose YOLO dataset
EXISTING_DATASET = (
    BASE_DIR
    / "dataset"
    / "nose_detection_yolo_tight"
)

TRAIN_IMAGES = (
    EXISTING_DATASET
    / "images"
    / "train"
)

TRAIN_LABELS = (
    EXISTING_DATASET
    / "labels"
    / "train"
)

VAL_IMAGES = (
    EXISTING_DATASET
    / "images"
    / "val"
)

VAL_LABELS = (
    EXISTING_DATASET
    / "labels"
    / "val"
)


# ============================================================
# SETTINGS
# ============================================================

TRAIN_RATIO = 0.80
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


# ============================================================
# CONVERT PIXEL BOX → YOLO FORMAT
# ============================================================

def convert_to_yolo(x, y, w, h, image_width, image_height):

    center_x = x + (w / 2)
    center_y = y + (h / 2)

    center_x /= image_width
    center_y /= image_height
    w /= image_width
    h /= image_height

    return center_x, center_y, w, h


# ============================================================
# MANUAL NOSE LABELING
# ============================================================

def label_images():

    print()
    print("=" * 70)
    print("             MANUAL NOSE LABELING")
    print("=" * 70)

    if not EXTRA_IMAGES.exists():
        print()
        print("ERROR:")
        print("Images folder not found:")
        print(EXTRA_IMAGES)
        return False

    EXTRA_LABELS.mkdir(
        parents=True,
        exist_ok=True
    )

    EXTRA_PREVIEWS.mkdir(
        parents=True,
        exist_ok=True
    )

    image_files = sorted([
        p
        for p in EXTRA_IMAGES.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_files:
        print()
        print("ERROR: No images found.")
        return False

    print()
    print(f"Images found: {len(image_files)}")

    print()
    print("INSTRUCTIONS")
    print("-" * 70)
    print("1. A dog image will appear.")
    print("2. Drag a rectangle around the DOG NOSE.")
    print("3. Press ENTER or SPACE to accept.")
    print("4. Press C to skip an image.")
    print("5. Do NOT include the whole face.")
    print("6. Draw the box tightly around the visible nose.")
    print()
    print("The coordinates will automatically be converted")
    print("to YOLO format and saved.")
    print()
    print("Press any key to begin...")
    input()

    labeled_count = 0
    skipped_count = 0

    for index, image_path in enumerate(image_files, start=1):

        print()
        print("=" * 70)
        print(
            f"IMAGE {index}/{len(image_files)}"
        )
        print(
            f"File: {image_path.name}"
        )
        print("=" * 70)

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print("Could not read image.")
            skipped_count += 1
            continue

        image_height, image_width = image.shape[:2]

        print()
        print("Draw a box around the DOG NOSE.")
        print("ENTER/SPACE = save")
        print("C = skip")

        roi = cv2.selectROI(
            "DRAW NOSE BOX",
            image,
            showCrosshair=True,
            fromCenter=False
        )

        cv2.destroyAllWindows()

        x, y, w, h = roi

        # ----------------------------------------------------
        # Skip
        # ----------------------------------------------------

        if w == 0 or h == 0:

            print(
                f"[SKIPPED] {image_path.name}"
            )

            skipped_count += 1
            continue

        # ----------------------------------------------------
        # Convert to YOLO
        # ----------------------------------------------------

        center_x, center_y, box_w, box_h = (
            convert_to_yolo(
                x,
                y,
                w,
                h,
                image_width,
                image_height
            )
        )

        # ----------------------------------------------------
        # Save YOLO label
        # Class 0 = dog_nose
        # ----------------------------------------------------

        label_path = (
            EXTRA_LABELS
            / f"{image_path.stem}.txt"
        )

        with open(
            label_path,
            "w"
        ) as f:

            f.write(
                f"0 "
                f"{center_x:.6f} "
                f"{center_y:.6f} "
                f"{box_w:.6f} "
                f"{box_h:.6f}\n"
            )

        # ----------------------------------------------------
        # Save preview
        # ----------------------------------------------------

        preview = image.copy()

        cv2.rectangle(
            preview,
            (int(x), int(y)),
            (int(x + w), int(y + h)),
            (0, 0, 255),
            3
        )

        cv2.putText(
            preview,
            "DOG NOSE",
            (int(x), max(30, int(y) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        preview_path = (
            EXTRA_PREVIEWS
            / image_path.name
        )

        cv2.imwrite(
            str(preview_path),
            preview
        )

        labeled_count += 1

        print()
        print(
            f"[LABELED] {image_path.name}"
        )

        print(
            f"Pixel box: "
            f"x={x}, y={y}, "
            f"w={w}, h={h}"
        )

        print(
            f"YOLO box: "
            f"{center_x:.6f} "
            f"{center_y:.6f} "
            f"{box_w:.6f} "
            f"{box_h:.6f}"
        )

    print()
    print("=" * 70)
    print("              LABELING COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Labeled : {labeled_count}"
    )

    print(
        f"Skipped : {skipped_count}"
    )

    print()
    print(
        "Labels saved to:"
    )
    print(EXTRA_LABELS)

    print()
    print(
        "Preview boxes saved to:"
    )
    print(EXTRA_PREVIEWS)

    return labeled_count > 0


# ============================================================
# COMBINE WITH EXISTING DATASET
# ============================================================

def combine_dataset():

    print()
    print("=" * 70)
    print("          COMBINING ANGLED NOSE DATASET")
    print("=" * 70)

    if not EXTRA_LABELS.exists():
        print()
        print("ERROR: Labels folder does not exist.")
        return

    # --------------------------------------------------------
    # Create destination folders
    # --------------------------------------------------------

    TRAIN_IMAGES.mkdir(
        parents=True,
        exist_ok=True
    )

    TRAIN_LABELS.mkdir(
        parents=True,
        exist_ok=True
    )

    VAL_IMAGES.mkdir(
        parents=True,
        exist_ok=True
    )

    VAL_LABELS.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find valid image-label pairs
    # --------------------------------------------------------

    image_files = sorted([
        p
        for p in EXTRA_IMAGES.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    valid_pairs = []

    for image_path in image_files:

        label_path = (
            EXTRA_LABELS
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():
            print(
                f"[NO LABEL] {image_path.name}"
            )
            continue

        if label_path.stat().st_size == 0:
            print(
                f"[EMPTY LABEL] {image_path.name}"
            )
            continue

        valid_pairs.append(
            (image_path, label_path)
        )

    print()
    print(
        f"Valid image-label pairs: "
        f"{len(valid_pairs)}"
    )

    if not valid_pairs:
        print()
        print("Nothing to combine.")
        return

    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    random.seed(
        RANDOM_SEED
    )

    random.shuffle(
        valid_pairs
    )

    # --------------------------------------------------------
    # 80 / 20 split
    # --------------------------------------------------------

    split_index = int(
        len(valid_pairs)
        * TRAIN_RATIO
    )

    train_pairs = valid_pairs[
        :split_index
    ]

    val_pairs = valid_pairs[
        split_index:
    ]

    print()
    print(
        f"New angled TRAIN: "
        f"{len(train_pairs)}"
    )

    print(
        f"New angled VAL  : "
        f"{len(val_pairs)}"
    )

    # --------------------------------------------------------
    # Copy TRAIN
    # --------------------------------------------------------

    print()
    print(
        "Adding angled images to TRAIN..."
    )

    for image_path, label_path in train_pairs:

        new_name = (
            "angled_"
            + image_path.name
        )

        destination_image = (
            TRAIN_IMAGES
            / new_name
        )

        destination_label = (
            TRAIN_LABELS
            / f"{Path(new_name).stem}.txt"
        )

        shutil.copy2(
            image_path,
            destination_image
        )

        shutil.copy2(
            label_path,
            destination_label
        )

        print(
            f"[TRAIN] {image_path.name}"
        )

    # --------------------------------------------------------
    # Copy VALIDATION
    # --------------------------------------------------------

    print()
    print(
        "Adding angled images to VALIDATION..."
    )

    for image_path, label_path in val_pairs:

        new_name = (
            "angled_"
            + image_path.name
        )

        destination_image = (
            VAL_IMAGES
            / new_name
        )

        destination_label = (
            VAL_LABELS
            / f"{Path(new_name).stem}.txt"
        )

        shutil.copy2(
            image_path,
            destination_image
        )

        shutil.copy2(
            label_path,
            destination_label
        )

        print(
            f"[VAL] {image_path.name}"
        )

    # --------------------------------------------------------
    # Final counts
    # --------------------------------------------------------

    final_train = [
        p
        for p in TRAIN_IMAGES.iterdir()
        if p.suffix.lower()
        in IMAGE_EXTENSIONS
    ]

    final_val = [
        p
        for p in VAL_IMAGES.iterdir()
        if p.suffix.lower()
        in IMAGE_EXTENSIONS
    ]

    print()
    print("=" * 70)
    print("             COMBINATION COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Angled images added : "
        f"{len(train_pairs) + len(val_pairs)}"
    )

    print(
        f"Added to train      : "
        f"{len(train_pairs)}"
    )

    print(
        f"Added to validation : "
        f"{len(val_pairs)}"
    )

    print()
    print(
        f"Final train images  : "
        f"{len(final_train)}"
    )

    print(
        f"Final val images    : "
        f"{len(final_val)}"
    )

    print()
    print(
        "Combined dataset:"
    )
    print(EXISTING_DATASET)

    print()
    print(
        "Original dataset files were kept."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    success = label_images()

    if success:

        print()
        print(
            "All labeling is finished."
        )

        print()
        input(
            "Press ENTER to combine the labeled images..."
        )

        combine_dataset()

    else:

        print()
        print(
            "No images were labeled."
        )