import os
import random
import shutil

BASE_DIR = r"E:\canine-nose-print-recognition\ai-model"

SOURCE_IMAGE_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "nose_detection",
    "images"
)

SOURCE_LABEL_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "nose_detection",
    "labels"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "nose_detection_yolo"
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

TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def main():

    print()
    print("=" * 70)
    print("             YOLO DATASET SPLIT")
    print("=" * 70)

    os.makedirs(TRAIN_IMAGE_DIR, exist_ok=True)
    os.makedirs(VAL_IMAGE_DIR, exist_ok=True)
    os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)
    os.makedirs(VAL_LABEL_DIR, exist_ok=True)

    image_files = [
        file
        for file in os.listdir(SOURCE_IMAGE_DIR)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    valid_pairs = []

    for image_file in image_files:

        image_name = os.path.splitext(
            image_file
        )[0]

        label_file = image_name + ".txt"

        label_path = os.path.join(
            SOURCE_LABEL_DIR,
            label_file
        )

        if os.path.exists(label_path):
            valid_pairs.append(
                (image_file, label_file)
            )

    print()
    print(
        f"Valid image-label pairs: {len(valid_pairs)}"
    )

    random.seed(RANDOM_SEED)
    random.shuffle(valid_pairs)

    split_index = int(
        len(valid_pairs) * TRAIN_RATIO
    )

    train_pairs = valid_pairs[:split_index]
    val_pairs = valid_pairs[split_index:]

    print(
        f"Training samples       : {len(train_pairs)}"
    )

    print(
        f"Validation samples     : {len(val_pairs)}"
    )

    for image_file, label_file in train_pairs:

        shutil.copy2(
            os.path.join(
                SOURCE_IMAGE_DIR,
                image_file
            ),
            os.path.join(
                TRAIN_IMAGE_DIR,
                image_file
            )
        )

        shutil.copy2(
            os.path.join(
                SOURCE_LABEL_DIR,
                label_file
            ),
            os.path.join(
                TRAIN_LABEL_DIR,
                label_file
            )
        )

    for image_file, label_file in val_pairs:

        shutil.copy2(
            os.path.join(
                SOURCE_IMAGE_DIR,
                image_file
            ),
            os.path.join(
                VAL_IMAGE_DIR,
                image_file
            )
        )

        shutil.copy2(
            os.path.join(
                SOURCE_LABEL_DIR,
                label_file
            ),
            os.path.join(
                VAL_LABEL_DIR,
                label_file
            )
        )

    print()
    print("Dataset created:")
    print(OUTPUT_DIR)

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()