import os
import json
import shutil
import random
import cv2


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"E:\canine-nose-print-recognition\ai-model"

DOGFLW_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "DogFLW",
    "train"
)

IMAGE_DIR = os.path.join(
    DOGFLW_DIR,
    "images"
)

LABEL_DIR = os.path.join(
    DOGFLW_DIR,
    "labels"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "face_detection_yolo"
)

TRAIN_IMAGE_DIR = os.path.join(
    OUTPUT_DIR,
    "images",
    "train"
)

VAL_IMAGE_DIR = os.path.join(
    OUTPUT_DIR,
    "images",
    "val"
)

TRAIN_LABEL_DIR = os.path.join(
    OUTPUT_DIR,
    "labels",
    "train"
)

VAL_LABEL_DIR = os.path.join(
    OUTPUT_DIR,
    "labels",
    "val"
)

PREVIEW_DIR = os.path.join(
    OUTPUT_DIR,
    "previews"
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
    ".bmp",
    ".webp"
}


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for directory in [
    TRAIN_IMAGE_DIR,
    VAL_IMAGE_DIR,
    TRAIN_LABEL_DIR,
    VAL_LABEL_DIR,
    PREVIEW_DIR
]:
    os.makedirs(directory, exist_ok=True)


# ============================================================
# FIND IMAGE FOR JSON FILE
# ============================================================

def find_image(json_file):

    stem = os.path.splitext(
        os.path.basename(json_file)
    )[0]

    for extension in IMAGE_EXTENSIONS:

        image_path = os.path.join(
            IMAGE_DIR,
            stem + extension
        )

        if os.path.exists(image_path):
            return image_path

    return None


# ============================================================
# CONVERT BOUNDING BOX TO YOLO FORMAT
# ============================================================

def convert_bbox_to_yolo(
    bbox,
    image_width,
    image_height
):

    # Convert coordinates to numbers
    try:
        x1, y1, x2, y2 = [
            float(value) for value in bbox
        ]
    except (ValueError, TypeError):
        return None

    # Clamp coordinates to image boundaries
    x1 = max(0.0, min(x1, float(image_width)))
    y1 = max(0.0, min(y1, float(image_height)))
    x2 = max(0.0, min(x2, float(image_width)))
    y2 = max(0.0, min(y2, float(image_height)))

    # Make sure coordinates are ordered correctly
    if x2 <= x1 or y2 <= y1:
        return None

    # Calculate center
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    # Calculate width and height
    width = x2 - x1
    height = y2 - y1

    # Normalize for YOLO
    center_x /= float(image_width)
    center_y /= float(image_height)

    width /= float(image_width)
    height /= float(image_height)

    return (
        center_x,
        center_y,
        width,
        height
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("          DOG FACE DETECTION DATASET PREPARATION")
    print("=" * 70)
    print()

    print(f"DogFLW images : {IMAGE_DIR}")
    print(f"DogFLW labels : {LABEL_DIR}")
    print(f"Output        : {OUTPUT_DIR}")
    print()

    json_files = [
        os.path.join(LABEL_DIR, file)
        for file in os.listdir(LABEL_DIR)
        if file.lower().endswith(".json")
    ]

    print(f"JSON files found: {len(json_files)}")
    print()

    samples = []
    skipped = 0

    # --------------------------------------------------------
    # READ JSON FILES
    # --------------------------------------------------------

    for json_file in json_files:

        image_path = find_image(json_file)

        if image_path is None:
            print(
                f"SKIP - image not found: "
                f"{os.path.basename(json_file)}"
            )
            skipped += 1
            continue

        try:

            with open(
                json_file,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

        except Exception as error:

            print(
                f"SKIP - invalid JSON: "
                f"{os.path.basename(json_file)}"
            )

            print(f"       {error}")

            skipped += 1
            continue

        # ----------------------------------------------------
        # GET FACE BOUNDING BOX
        # ----------------------------------------------------

        bbox = data.get("bounding_boxes")

        if bbox is None:

            print(
                f"SKIP - no bounding_boxes: "
                f"{os.path.basename(json_file)}"
            )

            skipped += 1
            continue

        # Handle possible nested bbox structure
        if (
            isinstance(bbox, list)
            and len(bbox) == 1
            and isinstance(bbox[0], list)
        ):
            bbox = bbox[0]

        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
        ):

            print(
                f"SKIP - invalid bounding box: "
                f"{os.path.basename(json_file)}"
            )

            skipped += 1
            continue

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image = cv2.imread(image_path)

        if image is None:

            print(
                f"SKIP - unreadable image: "
                f"{os.path.basename(image_path)}"
            )

            skipped += 1
            continue

        image_height, image_width = image.shape[:2]

        yolo_bbox = convert_bbox_to_yolo(
            bbox,
            image_width,
            image_height
        )

        if yolo_bbox is None:

            print(
                f"SKIP - invalid face box: "
                f"{os.path.basename(json_file)}"
            )

            skipped += 1
            continue

        samples.append(
            {
                "image": image_path,
                "bbox": yolo_bbox
            }
        )

    # --------------------------------------------------------
    # SHUFFLE
    # --------------------------------------------------------

    random.seed(RANDOM_SEED)
    random.shuffle(samples)

    split_index = int(
        len(samples) * TRAIN_RATIO
    )

    train_samples = samples[:split_index]
    val_samples = samples[split_index:]

    print()
    print(f"Valid samples     : {len(samples)}")
    print(f"Skipped samples   : {skipped}")
    print(f"Training samples  : {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print()

    # --------------------------------------------------------
    # COPY IMAGES + CREATE YOLO LABELS
    # --------------------------------------------------------

    def process_split(
        split_samples,
        image_output_dir,
        label_output_dir,
        split_name
    ):

        for index, sample in enumerate(split_samples):

            image_path = sample["image"]
            bbox = sample["bbox"]

            filename = os.path.basename(image_path)

            stem = os.path.splitext(filename)[0]

            output_image = os.path.join(
                image_output_dir,
                filename
            )

            output_label = os.path.join(
                label_output_dir,
                stem + ".txt"
            )

            shutil.copy2(
                image_path,
                output_image
            )

            # Class 0 = dog_face
            center_x, center_y, width, height = bbox

            with open(
                output_label,
                "w",
                encoding="utf-8"
            ) as file:

                file.write(
                    f"0 {center_x:.6f} "
                    f"{center_y:.6f} "
                    f"{width:.6f} "
                    f"{height:.6f}\n"
                )

            # ------------------------------------------------
            # PREVIEW
            # ------------------------------------------------

            image = cv2.imread(image_path)

            if image is not None:

                image_height, image_width = image.shape[:2]

                x_center = center_x * image_width
                y_center = center_y * image_height

                box_width = width * image_width
                box_height = height * image_height

                x1 = int(
                    x_center - box_width / 2
                )

                y1 = int(
                    y_center - box_height / 2
                )

                x2 = int(
                    x_center + box_width / 2
                )

                y2 = int(
                    y_center + box_height / 2
                )

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3
                )

                preview_name = (
                    f"{split_name}_{index:04d}_{filename}"
                )

                preview_path = os.path.join(
                    PREVIEW_DIR,
                    preview_name
                )

                cv2.imwrite(
                    preview_path,
                    image
                )

    process_split(
        train_samples,
        TRAIN_IMAGE_DIR,
        TRAIN_LABEL_DIR,
        "train"
    )

    process_split(
        val_samples,
        VAL_IMAGE_DIR,
        VAL_LABEL_DIR,
        "val"
    )

    # --------------------------------------------------------
    # CREATE DATA.YAML
    # --------------------------------------------------------

    data_yaml_path = os.path.join(
        OUTPUT_DIR,
        "data.yaml"
    )

    with open(
        data_yaml_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            f"path: {OUTPUT_DIR.replace(os.sep, '/')}\n"
        )

        file.write(
            "train: images/train\n"
        )

        file.write(
            "val: images/val\n"
        )

        file.write(
            "names:\n"
        )

        file.write(
            "  0: dog_face\n"
        )

    print("=" * 70)
    print("          DATASET PREPARATION COMPLETE")
    print("=" * 70)
    print()

    print(f"Dataset : {OUTPUT_DIR}")
    print(f"YAML   : {data_yaml_path}")
    print(f"Preview: {PREVIEW_DIR}")
    print()


if __name__ == "__main__":
    main()