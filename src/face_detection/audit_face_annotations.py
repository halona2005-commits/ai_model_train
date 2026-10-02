import os
import json

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


IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
]


def find_image(stem):
    matches = []

    for ext in IMAGE_EXTENSIONS:
        path = os.path.join(IMAGE_DIR, stem + ext)

        if os.path.exists(path):
            matches.append(path)

    return matches


def main():

    print()
    print("=" * 70)
    print("             DOGFLW FACE ANNOTATION AUDIT")
    print("=" * 70)

    json_files = [
        f for f in os.listdir(LABEL_DIR)
        if f.lower().endswith(".json")
    ]

    print(f"\nJSON files found: {len(json_files)}")

    valid = 0
    invalid_bbox = 0
    landmark_bbox_mismatch = 0
    missing_images = 0
    duplicate_images = 0

    suspicious = []

    for json_file in json_files:

        json_path = os.path.join(
            LABEL_DIR,
            json_file
        )

        stem = os.path.splitext(json_file)[0]

        # --------------------------------------------------
        # FIND IMAGE
        # --------------------------------------------------

        image_matches = find_image(stem)

        if len(image_matches) == 0:
            missing_images += 1

            suspicious.append({
                "file": json_file,
                "reason": "MISSING_IMAGE"
            })

            continue

        if len(image_matches) > 1:
            duplicate_images += 1

            suspicious.append({
                "file": json_file,
                "reason": "MULTIPLE_IMAGES",
                "images": image_matches
            })

            continue

        # --------------------------------------------------
        # READ JSON
        # --------------------------------------------------

        try:
            with open(
                json_path,
                "r",
                encoding="utf-8"
            ) as f:
                data = json.load(f)

        except Exception as e:

            suspicious.append({
                "file": json_file,
                "reason": f"JSON_ERROR: {e}"
            })

            continue

        # --------------------------------------------------
        # READ BOUNDING BOX
        # --------------------------------------------------

        bbox = data.get("bounding_boxes")

        if not bbox or len(bbox) != 4:

            invalid_bbox += 1

            suspicious.append({
                "file": json_file,
                "reason": "INVALID_BBOX"
            })

            continue

        try:
            x1, y1, x2, y2 = [
                float(v) for v in bbox
            ]

        except (ValueError, TypeError):

            invalid_bbox += 1

            suspicious.append({
                "file": json_file,
                "reason": "NON_NUMERIC_BBOX"
            })

            continue

        if x2 <= x1 or y2 <= y1:

            invalid_bbox += 1

            suspicious.append({
                "file": json_file,
                "reason": "INVALID_BBOX_COORDINATES",
                "bbox": [x1, y1, x2, y2]
            })

            continue

        # --------------------------------------------------
        # CHECK LANDMARKS
        # --------------------------------------------------

        landmarks = data.get("landmarks")

        if not landmarks or len(landmarks) != 46:

            suspicious.append({
                "file": json_file,
                "reason": "INVALID_LANDMARK_COUNT"
            })

            continue

        inside_count = 0

        for point in landmarks:

            try:
                px = float(point[0])
                py = float(point[1])

            except (ValueError, TypeError, IndexError):

                continue

            if (
                x1 <= px <= x2
                and
                y1 <= py <= y2
            ):
                inside_count += 1

        landmark_ratio = inside_count / len(landmarks)

        # --------------------------------------------------
        # FLAG BAD FACE BOX
        # --------------------------------------------------

        # A proper face bounding box should contain
        # essentially all face landmarks.

        if landmark_ratio < 0.90:

            landmark_bbox_mismatch += 1

            suspicious.append({
                "file": json_file,
                "reason": "LANDMARK_BBOX_MISMATCH",
                "bbox": [x1, y1, x2, y2],
                "landmarks_inside": inside_count,
                "landmarks_total": len(landmarks),
                "ratio": round(landmark_ratio, 3),
                "image": image_matches[0]
            })

        valid += 1

    # ------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    print(f"Total JSON files           : {len(json_files)}")
    print(f"Processed valid JSON       : {valid}")
    print(f"Invalid bounding boxes     : {invalid_bbox}")
    print(f"Missing images             : {missing_images}")
    print(f"Multiple matching images   : {duplicate_images}")
    print(
        f"Landmark/bbox mismatches   : "
        f"{landmark_bbox_mismatch}"
    )

    print()
    print("=" * 70)
    print("SUSPICIOUS SAMPLES")
    print("=" * 70)

    for item in suspicious[:50]:

        print()

        for key, value in item.items():
            print(f"{key}: {value}")

    print()

    if len(suspicious) > 50:
        print(
            f"... and {len(suspicious) - 50} more suspicious samples."
        )

    print()
    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()