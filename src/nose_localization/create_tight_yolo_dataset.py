import os
import shutil
import random


BASE_DIR = r"E:\canine-nose-print-recognition\ai-model"

SOURCE_IMAGES = os.path.join(
    BASE_DIR,
    "dataset",
    "nose_detection",
    "images"
)

SOURCE_LABELS = os.path.join(
    BASE_DIR,
    "dataset",
    "nose_detection",
    "labels"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "nose_detection_yolo_tight"
)


TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def main():

    print()
    print("=" * 70)
    print("       CREATING TIGHT YOLO DATASET")
    print("=" * 70)

    image_files = [
        f for f in os.listdir(SOURCE_IMAGES)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    valid_pairs = []

    for image_file in image_files:

        name = os.path.splitext(image_file)[0]
        label_file = name + ".txt"

        image_path = os.path.join(
            SOURCE_IMAGES,
            image_file
        )

        label_path = os.path.join(
            SOURCE_LABELS,
            label_file
        )

        if os.path.exists(label_path):
            valid_pairs.append(
                (image_path, label_path, image_file, label_file)
            )

    print()
    print(f"Valid image-label pairs: {len(valid_pairs)}")

    random.seed(RANDOM_SEED)
    random.shuffle(valid_pairs)

    split_index = int(
        len(valid_pairs) * TRAIN_RATIO
    )

    train_data = valid_pairs[:split_index]
    val_data = valid_pairs[split_index:]

    print(f"Training samples       : {len(train_data)}")
    print(f"Validation samples     : {len(val_data)}")

    train_images = os.path.join(
        OUTPUT_DIR,
        "images",
        "train"
    )

    val_images = os.path.join(
        OUTPUT_DIR,
        "images",
        "val"
    )

    train_labels = os.path.join(
        OUTPUT_DIR,
        "labels",
        "train"
    )

    val_labels = os.path.join(
        OUTPUT_DIR,
        "labels",
        "val"
    )

    for directory in [
        train_images,
        val_images,
        train_labels,
        val_labels
    ]:
        os.makedirs(
            directory,
            exist_ok=True
        )

    def copy_dataset(data, image_dir, label_dir):

        for image_path, label_path, image_file, label_file in data:

            shutil.copy2(
                image_path,
                os.path.join(
                    image_dir,
                    image_file
                )
            )

            shutil.copy2(
                label_path,
                os.path.join(
                    label_dir,
                    label_file
                ))

    copy_dataset(
        train_data,
        train_images,
        train_labels
    )

    copy_dataset(
        val_data,
        val_images,
        val_labels
    )

    data_yaml = os.path.join(
        OUTPUT_DIR,
        "data.yaml"
    )

    with open(
        data_yaml,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"""path: {OUTPUT_DIR.replace(os.sep, '/')}
train: images/train
val: images/val

names:
  0: dog_nose
"""
        )

    print()
    print("=" * 70)
    print("                 DATASET READY")
    print("=" * 70)

    print()
    print(f"Output directory:")
    print(OUTPUT_DIR)

    print()
    print("data.yaml:")
    print(data_yaml)

    print()
    print(f"Train: {len(train_data)}")
    print(f"Val  : {len(val_data)}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()