import os
import random
import cv2


BASE_DIR = r"E:\canine-nose-print-recognition\ai-model"

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "face_detection_yolo"
)

TRAIN_IMAGE_DIR = os.path.join(
    DATASET_DIR,
    "images",
    "train"
)

TRAIN_LABEL_DIR = os.path.join(
    DATASET_DIR,
    "labels",
    "train"
)

VAL_IMAGE_DIR = os.path.join(
    DATASET_DIR,
    "images",
    "val"
)

VAL_LABEL_DIR = os.path.join(
    DATASET_DIR,
    "labels",
    "val"
)

PREVIEW_DIR = os.path.join(
    DATASET_DIR,
    "sanity_previews"
)

NUM_TRAIN_PREVIEWS = 15
NUM_VAL_PREVIEWS = 10

random.seed(42)


def draw_yolo_box(image, label_path):

    height, width = image.shape[:2]

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        lines = f.readlines()

    for line in lines:

        parts = line.strip().split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])

        center_x = float(parts[1])
        center_y = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])

        # YOLO normalized coordinates
        x_center = center_x * width
        y_center = center_y * height
        w = box_width * width
        h = box_height * height

        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3
        )

        cv2.putText(
            image,
            "dog_face",
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    return image


def create_previews(
    image_dir,
    label_dir,
    output_dir,
    count,
    prefix
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        )
    ]

    random.shuffle(image_files)

    selected = image_files[:count]

    created = 0

    for image_file in selected:

        image_path = os.path.join(
            image_dir,
            image_file
        )

        stem = os.path.splitext(
            image_file
        )[0]

        label_path = os.path.join(
            label_dir,
            stem + ".txt"
        )

        if not os.path.exists(label_path):
            continue

        image = cv2.imread(image_path)

        if image is None:
            continue

        image = draw_yolo_box(
            image,
            label_path
        )

        output_path = os.path.join(
            output_dir,
            f"{prefix}_{stem}.png"
        )

        cv2.imwrite(
            output_path,
            image
        )

        created += 1

    return created


def main():

    print()
    print("=" * 70)
    print("             YOLO FACE DATASET SANITY CHECK")
    print("=" * 70)

    os.makedirs(
        PREVIEW_DIR,
        exist_ok=True
    )

    train_count = create_previews(
        TRAIN_IMAGE_DIR,
        TRAIN_LABEL_DIR,
        PREVIEW_DIR,
        NUM_TRAIN_PREVIEWS,
        "train"
    )

    val_count = create_previews(
        VAL_IMAGE_DIR,
        VAL_LABEL_DIR,
        PREVIEW_DIR,
        NUM_VAL_PREVIEWS,
        "val"
    )

    print()
    print(f"Training previews   : {train_count}")
    print(f"Validation previews : {val_count}")

    print()
    print("Preview folder:")
    print(PREVIEW_DIR)

    print()
    print("=" * 70)
    print("                 CHECK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()