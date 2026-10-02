import os
import json
import shutil
import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"E:\canine-nose-print-recognition\ai-model"

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "DogFLW"
)

IMAGE_DIR = os.path.join(
    DATASET_DIR,
    "train",
    "images"
)

LABEL_DIR = os.path.join(
    DATASET_DIR,
    "train",
    "labels"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "nose_detection"
)


# ============================================================
# SETTINGS
# ============================================================

NOSE_POINTS = [
    25,
    36,
    37,
    38,
    39
]

# Extra margin around the landmark-derived nose box.
# This prevents the detector from learning an unrealistically
# tiny box around only the landmark points.

NOSE_MARGIN = 0.15

# Minimum allowed nose box size in pixels after calculation.
MIN_NOSE_SIZE = 8

# Face crop margin.
# DogFLW face boxes already contain approximately 10% surrounding
# space, so we don't add a large additional margin.

FACE_MARGIN = 0.05


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

FACE_IMAGES_DIR = os.path.join(
    OUTPUT_DIR,
    "images"
)

FACE_LABELS_DIR = os.path.join(
    OUTPUT_DIR,
    "labels"
)

PREVIEW_DIR = os.path.join(
    OUTPUT_DIR,
    "previews"
)


for directory in [
    FACE_IMAGES_DIR,
    FACE_LABELS_DIR,
    PREVIEW_DIR
]:

    os.makedirs(
        directory,
        exist_ok=True
    )


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(image_name):

    extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".JPG",
        ".JPEG",
        ".PNG"
    ]

    for extension in extensions:

        image_path = os.path.join(
            IMAGE_DIR,
            image_name + extension
        )

        if os.path.exists(image_path):

            return image_path

    return None


# ============================================================
# CLAMP VALUE
# ============================================================

def clamp(
    value,
    minimum,
    maximum
):

    return max(
        minimum,
        min(value, maximum)
    )


# ============================================================
# CREATE NOSE BOX
# ============================================================

def create_nose_box(
    landmarks,
    width,
    height
):

    points = []

    for index in NOSE_POINTS:

        point = landmarks[index]

        x = float(point[0])
        y = float(point[1])

        if not (
            np.isfinite(x)
            and np.isfinite(y)
        ):

            return None

        points.append(
            (x, y)
        )


    xs = [
        point[0]
        for point in points
    ]

    ys = [
        point[1]
        for point in points
    ]


    # --------------------------------------------------------
    # Raw landmark bounding box
    # --------------------------------------------------------

    x1 = min(xs)
    y1 = min(ys)

    x2 = max(xs)
    y2 = max(ys)


    box_width = x2 - x1
    box_height = y2 - y1


    # --------------------------------------------------------
    # If landmarks are extremely close,
    # give them a minimum box size.
    # --------------------------------------------------------

    box_width = max(
        box_width,
        MIN_NOSE_SIZE
    )

    box_height = max(
        box_height,
        MIN_NOSE_SIZE
    )


    # --------------------------------------------------------
    # Add adaptive margin
    # --------------------------------------------------------

    margin_x = box_width * NOSE_MARGIN
    margin_y = box_height * NOSE_MARGIN


    x1 = x1 - margin_x
    y1 = y1 - margin_y

    x2 = x2 + margin_x
    y2 = y2 + margin_y


    # --------------------------------------------------------
    # Clamp to image
    # --------------------------------------------------------

    x1 = clamp(
        x1,
        0,
        width - 1
    )

    y1 = clamp(
        y1,
        0,
        height - 1
    )

    x2 = clamp(
        x2,
        1,
        width
    )

    y2 = clamp(
        y2,
        1,
        height
    )


    if x2 <= x1 or y2 <= y1:

        return None


    return (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# YOLO BOX CONVERSION
# ============================================================

def to_yolo_box(
    x1,
    y1,
    x2,
    y2,
    width,
    height
):

    box_width = x2 - x1
    box_height = y2 - y1

    center_x = (
        x1 + x2
    ) / 2

    center_y = (
        y1 + y2
    ) / 2


    return (
        center_x / width,
        center_y / height,
        box_width / width,
        box_height / height
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("       NOSE DETECTION DATASET PREPARATION")
    print("=" * 70)


    json_files = [
        file
        for file in os.listdir(
            LABEL_DIR
        )
        if file.endswith(".json")
    ]


    print()
    print(
        f"JSON files found: {len(json_files)}"
    )


    valid = 0
    skipped = 0

    preview_count = 0


    # ========================================================
    # PROCESS
    # ========================================================

    for count, json_file in enumerate(
        json_files,
        start=1
    ):

        image_name = os.path.splitext(
            json_file
        )[0]


        # ----------------------------------------------------
        # Find image
        # ----------------------------------------------------

        image_path = find_image(
            image_name
        )


        if image_path is None:

            skipped += 1
            continue


        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = cv2.imread(
            image_path
        )


        if image is None:

            skipped += 1
            continue


        height, width = image.shape[:2]


        # ----------------------------------------------------
        # Load JSON
        # ----------------------------------------------------

        json_path = os.path.join(
            LABEL_DIR,
            json_file
        )


        try:

            with open(
                json_path,
                "r"
            ) as file:

                data = json.load(file)

        except Exception:

            skipped += 1
            continue


        landmarks = data.get(
            "landmarks"
        )

        bounding_boxes = data.get(
            "bounding_boxes"
        )


        if landmarks is None:

            skipped += 1
            continue


        if bounding_boxes is None:

            skipped += 1
            continue


        if len(landmarks) <= max(
            NOSE_POINTS
        ):

            skipped += 1
            continue


        # ====================================================
        # DOG FACE BOUNDING BOX
        # ====================================================

        try:

            face_box = [
                float(value)
                for value in bounding_boxes
            ]

        except Exception:

            skipped += 1
            continue


        if len(face_box) != 4:

            skipped += 1
            continue


        face_x1 = face_box[0]
        face_y1 = face_box[1]
        face_x2 = face_box[2]
        face_y2 = face_box[3]


        # ----------------------------------------------------
        # Validate face box
        # ----------------------------------------------------

        if not all(
            np.isfinite(value)
            for value in face_box
        ):

            skipped += 1
            continue


        face_x1 = clamp(
            face_x1,
            0,
            width - 1
        )

        face_y1 = clamp(
            face_y1,
            0,
            height - 1
        )

        face_x2 = clamp(
            face_x2,
            1,
            width
        )

        face_y2 = clamp(
            face_y2,
            1,
            height
        )


        if (
            face_x2 <= face_x1
            or
            face_y2 <= face_y1
        ):

            skipped += 1
            continue


        # ====================================================
        # ADD SMALL FACE MARGIN
        # ====================================================

        face_width = (
            face_x2 - face_x1
        )

        face_height = (
            face_y2 - face_y1
        )


        margin_x = (
            face_width *
            FACE_MARGIN
        )

        margin_y = (
            face_height *
            FACE_MARGIN
        )


        face_x1 = clamp(
            face_x1 - margin_x,
            0,
            width - 1
        )

        face_y1 = clamp(
            face_y1 - margin_y,
            0,
            height - 1
        )

        face_x2 = clamp(
            face_x2 + margin_x,
            1,
            width
        )

        face_y2 = clamp(
            face_y2 + margin_y,
            1,
            height
        )


        # ====================================================
        # CROP FACE
        # ====================================================

        face_x1_int = int(
            round(face_x1)
        )

        face_y1_int = int(
            round(face_y1)
        )

        face_x2_int = int(
            round(face_x2)
        )

        face_y2_int = int(
            round(face_y2)
        )


        face_crop = image[
            face_y1_int:face_y2_int,
            face_x1_int:face_x2_int
        ]


        if face_crop.size == 0:

            skipped += 1
            continue


        crop_height, crop_width = (
            face_crop.shape[:2]
        )


        # ====================================================
        # CONVERT NOSE LANDMARKS TO FACE-CROP COORDINATES
        # ====================================================

        crop_landmarks = []


        valid_landmarks = True


        for index in NOSE_POINTS:

            point = landmarks[index]

            x = float(point[0])
            y = float(point[1])


            if not (
                np.isfinite(x)
                and np.isfinite(y)
            ):

                valid_landmarks = False
                break


            crop_x = (
                x - face_x1_int
            )

            crop_y = (
                y - face_y1_int
            )


            crop_landmarks.append(
                (
                    crop_x,
                    crop_y
                )
            )


        if not valid_landmarks:

            skipped += 1
            continue


        # ====================================================
        # CREATE NOSE BOX INSIDE FACE CROP
        # ====================================================

        xs = [
            point[0]
            for point in crop_landmarks
        ]

        ys = [
            point[1]
            for point in crop_landmarks
        ]


        nose_x1 = min(xs)
        nose_y1 = min(ys)

        nose_x2 = max(xs)
        nose_y2 = max(ys)


        nose_width = (
            nose_x2 - nose_x1
        )

        nose_height = (
            nose_y2 - nose_y1
        )


        nose_width = max(
            nose_width,
            MIN_NOSE_SIZE
        )

        nose_height = max(
            nose_height,
            MIN_NOSE_SIZE
        )


        margin_x = (
            nose_width *
            NOSE_MARGIN
        )

        margin_y = (
            nose_height *
            NOSE_MARGIN
        )


        nose_x1 -= margin_x
        nose_y1 -= margin_y

        nose_x2 += margin_x
        nose_y2 += margin_y


        # Clamp to face crop

        nose_x1 = clamp(
            nose_x1,
            0,
            crop_width - 1
        )

        nose_y1 = clamp(
            nose_y1,
            0,
            crop_height - 1
        )

        nose_x2 = clamp(
            nose_x2,
            1,
            crop_width
        )

        nose_y2 = clamp(
            nose_y2,
            1,
            crop_height
        )


        if (
            nose_x2 <= nose_x1
            or
            nose_y2 <= nose_y1
        ):

            skipped += 1
            continue


        # ====================================================
        # CONVERT TO YOLO FORMAT
        # ====================================================

        center_x = (
            nose_x1 + nose_x2
        ) / 2

        center_y = (
            nose_y1 + nose_y2
        ) / 2

        box_width = (
            nose_x2 - nose_x1
        )

        box_height = (
            nose_y2 - nose_y1
        )


        yolo_x = (
            center_x /
            crop_width
        )

        yolo_y = (
            center_y /
            crop_height
        )

        yolo_width = (
            box_width /
            crop_width
        )

        yolo_height = (
            box_height /
            crop_height
        )


        # ====================================================
        # SAVE FACE IMAGE
        # ====================================================

        output_image_name = (
            image_name + ".jpg"
        )

        output_label_name = (
            image_name + ".txt"
        )


        output_image_path = os.path.join(
            FACE_IMAGES_DIR,
            output_image_name
        )

        output_label_path = os.path.join(
            FACE_LABELS_DIR,
            output_label_name
        )


        cv2.imwrite(
            output_image_path,
            face_crop
        )


        # Class 0 = nose

        with open(
            output_label_path,
            "w"
        ) as file:

            file.write(
                f"0 "
                f"{yolo_x:.6f} "
                f"{yolo_y:.6f} "
                f"{yolo_width:.6f} "
                f"{yolo_height:.6f}\n"
            )


        # ====================================================
        # PREVIEW FIRST 20
        # ====================================================

        if preview_count < 20:

            preview = face_crop.copy()


            cv2.rectangle(
                preview,
                (
                    int(nose_x1),
                    int(nose_y1)
                ),
                (
                    int(nose_x2),
                    int(nose_y2)
                ),
                (0, 0, 255),
                2
            )


            cv2.putText(
                preview,
                "NOSE",
                (
                    int(nose_x1),
                    max(
                        20,
                        int(nose_y1) - 5
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )


            preview_path = os.path.join(
                PREVIEW_DIR,
                f"{preview_count:02d}_{image_name}.jpg"
            )


            cv2.imwrite(
                preview_path,
                preview
            )


            preview_count += 1


        valid += 1


        if count % 500 == 0:

            print(
                f"Processed {count} files..."
            )


    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("                    RESULTS")
    print("=" * 70)

    print()
    print(
        f"Valid samples : {valid}"
    )

    print(
        f"Skipped       : {skipped}"
    )

    print()
    print(
        "Face crops:"
    )

    print(
        FACE_IMAGES_DIR
    )

    print()
    print(
        "YOLO labels:"
    )

    print(
        FACE_LABELS_DIR
    )

    print()
    print(
        "Preview images:"
    )

    print(
        PREVIEW_DIR
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()