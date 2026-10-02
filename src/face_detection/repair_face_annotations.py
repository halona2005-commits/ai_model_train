import os
import json
import cv2


BASE_DIR = r"E:\canine-nose-print-recognition\ai-model"

LABEL_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "DogFLW",
    "train",
    "labels"
)

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "DogFLW",
    "train",
    "images"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "face_annotation_repair"
)

REPAIRED_LABEL_DIR = os.path.join(
    OUTPUT_DIR,
    "labels"
)

PREVIEW_DIR = os.path.join(
    OUTPUT_DIR,
    "previews"
)

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
]

# If fewer than 90% of landmarks are inside
# the original box, rebuild the box.
LANDMARK_THRESHOLD = 0.90

# DogFLW describes the face box as having
# approximately 10% surrounding space.
MARGIN = 0.08


def find_image(stem):

    for ext in IMAGE_EXTENSIONS:

        path = os.path.join(
            IMAGE_DIR,
            stem + ext
        )

        if os.path.exists(path):
            return path

    return None


def get_landmark_box(landmarks):

    points = []

    for point in landmarks:

        if len(point) < 2:
            continue

        try:
            x = float(point[0])
            y = float(point[1])
        except (ValueError, TypeError):
            continue

        points.append((x, y))

    if len(points) < 4:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    return (
        min(xs),
        min(ys),
        max(xs),
        max(ys)
    )


def add_margin(
    bbox,
    image_width,
    image_height
):

    x1, y1, x2, y2 = bbox

    width = x2 - x1
    height = y2 - y1

    margin_x = width * MARGIN
    margin_y = height * MARGIN

    x1 -= margin_x
    y1 -= margin_y
    x2 += margin_x
    y2 += margin_y

    x1 = max(0.0, min(x1, image_width))
    y1 = max(0.0, min(y1, image_height))
    x2 = max(0.0, min(x2, image_width))
    y2 = max(0.0, min(y2, image_height))

    return (
        x1,
        y1,
        x2,
        y2
    )


def landmarks_inside_ratio(
    landmarks,
    bbox
):

    x1, y1, x2, y2 = bbox

    valid = 0
    inside = 0

    for point in landmarks:

        if len(point) < 2:
            continue

        try:
            x = float(point[0])
            y = float(point[1])
        except (ValueError, TypeError):
            continue

        valid += 1

        if (
            x1 <= x <= x2
            and
            y1 <= y <= y2
        ):
            inside += 1

    if valid == 0:
        return 0.0

    return inside / valid


def draw_box(
    image,
    bbox,
    text
):

    x1, y1, x2, y2 = [
        int(round(v))
        for v in bbox
    ]

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        3
    )

    cv2.putText(
        image,
        text,
        (x1, max(25, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )


def main():

    os.makedirs(
        REPAIRED_LABEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        PREVIEW_DIR,
        exist_ok=True
    )

    json_files = [
        f
        for f in os.listdir(LABEL_DIR)
        if f.lower().endswith(".json")
    ]

    kept = 0
    repaired = 0
    skipped = 0

    print()
    print("=" * 70)
    print("          DOGFLW FACE ANNOTATION REPAIR")
    print("=" * 70)

    print()
    print(f"JSON files: {len(json_files)}")

    for index, json_file in enumerate(
        json_files,
        start=1
    ):

        json_path = os.path.join(
            LABEL_DIR,
            json_file
        )

        stem = os.path.splitext(
            json_file
        )[0]

        image_path = find_image(stem)

        if image_path is None:
            skipped += 1
            continue

        image = cv2.imread(image_path)

        if image is None:
            skipped += 1
            continue

        image_height, image_width = image.shape[:2]

        try:

            with open(
                json_path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

        except Exception:

            skipped += 1
            continue

        landmarks = data.get("landmarks")

        if not landmarks:
            skipped += 1
            continue

        landmark_box = get_landmark_box(
            landmarks
        )

        if landmark_box is None:
            skipped += 1
            continue

        # --------------------------------------------------
        # CHECK ORIGINAL BOUNDING BOX
        # --------------------------------------------------

        original_box = data.get(
            "bounding_boxes"
        )

        original_valid = False

        if (
            isinstance(original_box, list)
            and
            len(original_box) == 4
        ):

            try:

                original_box = [
                    float(v)
                    for v in original_box
                ]

                x1, y1, x2, y2 = original_box

                if (
                    x2 > x1
                    and
                    y2 > y1
                ):
                    original_valid = True

            except (
                ValueError,
                TypeError
            ):
                original_valid = False

        # --------------------------------------------------
        # DECIDE KEEP OR REPAIR
        # --------------------------------------------------

        ratio = 0.0

        if original_valid:

            ratio = landmarks_inside_ratio(
                landmarks,
                original_box
            )

        # Build the final face box from the 46 landmarks.
        # This gives us a consistent, slightly tight face crop.

        final_box = add_margin(
            landmark_box,
            image_width,
            image_height
        )

        if (
            original_valid
            and
            ratio >= LANDMARK_THRESHOLD
        ):
            status = "TIGHTENED"
            kept += 1
        else:
            status = "REPAIRED"
            repaired += 1

        # --------------------------------------------------
        # SAVE NEW JSON
        # --------------------------------------------------

        output_data = dict(data)

        output_data["bounding_boxes"] = [
            round(v, 2)
            for v in final_box
        ]

        output_data["face_box_source"] = status

        output_data[
            "original_landmark_inside_ratio"
        ] = round(ratio, 3)

        output_json_path = os.path.join(
            REPAIRED_LABEL_DIR,
            json_file
        )

        with open(
            output_json_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                output_data,
                f,
                indent=2
            )

        # --------------------------------------------------
        # CREATE PREVIEW
        # --------------------------------------------------

        preview = image.copy()

        draw_box(
            preview,
            final_box,
            status
        )

        preview_path = os.path.join(
            PREVIEW_DIR,
            stem + ".png"
        )

        cv2.imwrite(
            preview_path,
            preview
        )

        if index % 250 == 0:

            print(
                f"Processed {index}/{len(json_files)}..."
            )

    print()
    print("=" * 70)
    print("                 REPAIR COMPLETE")
    print("=" * 70)

    print()
    print(f"Original boxes kept : {kept}")
    print(f"Boxes repaired      : {repaired}")
    print(f"Skipped             : {skipped}")

    print()
    print("Repaired labels:")
    print(REPAIRED_LABEL_DIR)

    print()
    print("Preview images:")
    print(PREVIEW_DIR)

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()