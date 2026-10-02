import shutil
import os
import json
import shutil
import random
import cv2


BASE_DIR = r"E:\canine-nose-print-recognition\ai-model"

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "DogFLW",
    "train",
    "images"
)

LABEL_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "face_annotation_repair",
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


IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
]

VAL_RATIO = 0.20
RANDOM_SEED = 42


def find_image(stem):

    for ext in IMAGE_EXTENSIONS:

        path = os.path.join(
            IMAGE_DIR,
            stem + ext
        )

        if os.path.exists(path):
            return path

    return None


def convert_to_yolo(
    bbox,
    image_width,
    image_height
):

    x1, y1, x2, y2 = bbox

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    width = x2 - x1
    height = y2 - y1

    center_x /= image_width
    center_y /= image_height
    width /= image_width
    height /= image_height

    return (
        center_x,
        center_y,
        width,
        height
    )


def main():

    print()
    print("=" * 70)
    print("          CREATE FINAL FACE YOLO DATASET")
    print("=" * 70)

    # --------------------------------------------------
    # REMOVE OLD DATASET
    # --------------------------------------------------

    if os.path.exists(OUTPUT_DIR):

        print()
        print("Removing previous YOLO dataset...")

        shutil.rmtree(OUTPUT_DIR)

        print("Old dataset removed.")

    # --------------------------------------------------
    # CREATE DIRECTORIES
    # --------------------------------------------------

    for directory in [
        TRAIN_IMAGE_DIR,
        VAL_IMAGE_DIR,
        TRAIN_LABEL_DIR,
        VAL_LABEL_DIR
    ]:

        os.makedirs(
            directory,
            exist_ok=True
        )

    # --------------------------------------------------
    # FIND JSON FILES
    # --------------------------------------------------

    json_files = [
        f
        for f in os.listdir(LABEL_DIR)
        if f.lower().endswith(".json")
    ]

    print()
    print(f"Verified annotations: {len(json_files)}")

    samples = []

    # --------------------------------------------------
    # PREPARE SAMPLES
    # --------------------------------------------------

    for json_file in json_files:

        stem = os.path.splitext(
            json_file
        )[0]

        image_path = find_image(stem)

        if image_path is None:
            continue

        json_path = os.path.join(
            LABEL_DIR,
            json_file
        )

        try:

            with open(
                json_path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            bbox = data["bounding_boxes"]

            bbox = [
                float(v)
                for v in bbox
            ]

        except Exception:

            continue

        image = cv2.imread(image_path)

        if image is None:
            continue

        image_height, image_width = image.shape[:2]

        if (
            bbox[2] <= bbox[0]
            or
            bbox[3] <= bbox[1]
        ):
            continue

        yolo_box = convert_to_yolo(
            bbox,
            image_width,
            image_height
        )

        samples.append(
            (
                stem,
                image_path,
                yolo_box
            )
        )

    # --------------------------------------------------
    # SHUFFLE + SPLIT
    # --------------------------------------------------

    random.seed(RANDOM_SEED)

    random.shuffle(samples)

    split_index = int(
        len(samples) * (1 - VAL_RATIO)
    )

    train_samples = samples[:split_index]
    val_samples = samples[split_index:]

    print()
    print(f"Total samples : {len(samples)}")
    print(f"Training      : {len(train_samples)}")
    print(f"Validation    : {len(val_samples)}")

    # --------------------------------------------------
    # COPY DATA
    # --------------------------------------------------

    def process_samples(
        sample_list,
        image_output_dir,
        label_output_dir
    ):

        count = 0

        for (
            stem,
            image_path,
            yolo_box
        ) in sample_list:

            image_filename = os.path.basename(
                image_path
            )

            output_image = os.path.join(
                image_output_dir,
                image_filename
            )

            output_label = os.path.join(
                label_output_dir,
                stem + ".txt"
            )

            shutil.copy2(
                image_path,
                output_image
            )

            center_x, center_y, width, height = yolo_box

            with open(
                output_label,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(
                    f"0 "
                    f"{center_x:.6f} "
                    f"{center_y:.6f} "
                    f"{width:.6f} "
                    f"{height:.6f}\n"
                )

            count += 1

        return count

    train_count = process_samples(
        train_samples,
        TRAIN_IMAGE_DIR,
        TRAIN_LABEL_DIR
    )

    val_count = process_samples(
        val_samples,
        VAL_IMAGE_DIR,
        VAL_LABEL_DIR
    )

    # --------------------------------------------------
    # CREATE DATA.YAML
    # --------------------------------------------------

    yaml_path = os.path.join(
        OUTPUT_DIR,
        "data.yaml"
    )

    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"path: {OUTPUT_DIR.replace(os.sep, '/')}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"names:\n"
            f"  0: dog_face\n"
        )

    print()
    print("=" * 70)
    print("                 DATASET COMPLETE")
    print("=" * 70)

    print()
    print(f"Training images : {train_count}")
    print(f"Validation images: {val_count}")

    print()
    print("Dataset:")
    print(OUTPUT_DIR)

    print()
    print("YAML:")
    print(yaml_path)

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()