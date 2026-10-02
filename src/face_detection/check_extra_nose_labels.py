import os
import cv2
from pathlib import Path


BASE_DIR = Path(
    r"E:\canine-nose-print-recognition\ai-model"
)

IMAGE_DIR = (
    BASE_DIR
    / "dataset"
    / "nose_detection_extra"
    / "images"
)

LABEL_DIR = (
    BASE_DIR
    / "dataset"
    / "nose_detection_extra"
    / "labels"
)

OUTPUT_DIR = (
    BASE_DIR
    / "dataset"
    / "nose_detection_extra"
    / "label_check"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    image_files = sorted([
        p
        for p in IMAGE_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    print()
    print("=" * 70)
    print("          CHECKING ANGLED NOSE LABELS")
    print("=" * 70)

    checked = 0

    for image_path in image_files:

        label_path = (
            LABEL_DIR
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():
            continue

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        height, width = image.shape[:2]

        with open(label_path, "r") as f:
            line = f.readline().strip()

        parts = line.split()

        if len(parts) != 5:
            print(
                f"[INVALID] {image_path.name}"
            )
            continue

        class_id = int(parts[0])

        center_x = float(parts[1])
        center_y = float(parts[2])
        box_w = float(parts[3])
        box_h = float(parts[4])

        # YOLO normalized → pixel coordinates

        x1 = int(
            (center_x - box_w / 2)
            * width
        )

        y1 = int(
            (center_y - box_h / 2)
            * height
        )

        x2 = int(
            (center_x + box_w / 2)
            * width
        )

        y2 = int(
            (center_y + box_h / 2)
            * height
        )

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        # Draw box

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3
        )

        cv2.putText(
            image,
            "DOG NOSE",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        output_path = (
            OUTPUT_DIR
            / image_path.name
        )

        cv2.imwrite(
            str(output_path),
            image
        )

        checked += 1

    print()
    print(
        f"Labels checked: {checked}"
    )

    print()
    print(
        "Checked images saved to:"
    )
    print(OUTPUT_DIR)

    print()
    print(
        "Open the label_check folder and inspect"
    )
    print(
        "the red boxes on all 36 images."
    )

    print()
    print(
        "Every red box should tightly cover the nose."
    )


if __name__ == "__main__":
    main()