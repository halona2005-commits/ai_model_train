import os
import shutil
import cv2
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"E:\canine-nose-print-recognition\ai-model"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "face_detector",
    "yolo11n_face_final",
    "weights",
    "best.pt"
)

INPUT_DIR = os.path.join(
    BASE_DIR,
    "random_dog"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "face_crops"
)


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE = 0.25

# Add 15% padding around detected face
PADDING = 0.15

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
)


# ============================================================
# CROP FACE
# ============================================================

def crop_face(image, box):

    height, width = image.shape[:2]

    x1, y1, x2, y2 = box.astype(int)

    face_width = x2 - x1
    face_height = y2 - y1

    # Add padding
    pad_x = int(face_width * PADDING)
    pad_y = int(face_height * PADDING)

    x1 = x1 - pad_x
    y1 = y1 - pad_y
    x2 = x2 + pad_x
    y2 = y2 + pad_y

    # Keep coordinates inside image
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2]


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("              DOG FACE CROPPING")
    print("=" * 70)

    print()
    print(f"Model   : {MODEL_PATH}")
    print(f"Input   : {INPUT_DIR}")
    print(f"Output  : {OUTPUT_DIR}")
    print(f"Padding : {PADDING * 100:.0f}%")

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        print()
        print("ERROR: Face detector model not found:")
        print(MODEL_PATH)
        return

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not os.path.exists(INPUT_DIR):

        print()
        print("ERROR: random_dog folder not found:")
        print(INPUT_DIR)
        return

    # --------------------------------------------------------
    # CLEAR OLD FACE CROPS
    # --------------------------------------------------------

    if os.path.exists(OUTPUT_DIR):

        print()
        print("Removing previous face crops...")

        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("Loading face detector...")

    model = YOLO(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    image_files = [
        f
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(
            IMAGE_EXTENSIONS
        )
    ]

    print()
    print(
        f"Found {len(image_files)} images."
    )

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for image_name in sorted(image_files):

        image_path = os.path.join(
            INPUT_DIR,
            image_name
        )

        image = cv2.imread(
            image_path
        )

        if image is None:

            print(
                f"[ERROR] Could not read "
                f"{image_name}"
            )

            continue

        # ----------------------------------------------------
        # Face detection
        # ----------------------------------------------------

        results = model.predict(
            source=image,
            imgsz=640,
            conf=CONFIDENCE,
            verbose=False
        )

        result = results[0]

        if (
            result.boxes is None
            or len(result.boxes) == 0
        ):

            print(
                f"[NO FACE] {image_name}"
            )

            continue

        # ----------------------------------------------------
        # Choose highest-confidence face
        # ----------------------------------------------------

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )

        best_index = confidences.argmax()

        confidence = float(
            confidences[best_index]
        )

        box = (
            result.boxes.xyxy[
                best_index
            ]
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Crop with padding
        # ----------------------------------------------------

        face_crop = crop_face(
            image,
            box
        )

        if face_crop is None:

            print(
                f"[INVALID CROP] "
                f"{image_name}"
            )

            continue

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        base_name = os.path.splitext(
            image_name
        )[0]

        output_name = (
            f"{base_name}_face.jpg"
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            output_name
        )

        cv2.imwrite(
            output_path,
            face_crop
        )

        print(
            f"[FACE] {image_name} "
            f"-> confidence={confidence:.2f} "
            f"-> {output_name}"
        )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("              FACE CROPPING COMPLETE")
    print("=" * 70)

    print()
    print("Face crops saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()