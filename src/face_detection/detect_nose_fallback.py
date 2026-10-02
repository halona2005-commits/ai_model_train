import os
import cv2
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"E:\canine-nose-print-recognition\ai-model"

BEST_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "nose_detector",
    "yolo11n_nose_tight",
    "weights",
    "best.pt"
)

LAST_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "nose_detector",
    "yolo11n_nose_augmented",
    "weights",
    "last.pt"
)

INPUT_DIR = os.path.join(
    BASE_DIR,
    "face_crops"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "nose_fallback_results"
)

# Best model is tried first
BEST_CONFIDENCE = 0.25

# Last model is the fallback
LAST_CONFIDENCE = 0.15

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
)


# ============================================================
# FIND BEST DETECTION
# ============================================================

def get_best_detection(result):

    if result.boxes is None:
        return None

    if len(result.boxes) == 0:
        return None

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

    return box, confidence


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path,
    best_model,
    last_model
):

    image = cv2.imread(
        image_path
    )

    if image is None:
        print(
            f"[ERROR] Could not read "
            f"{os.path.basename(image_path)}"
        )
        return

    height, width = image.shape[:2]

    # --------------------------------------------------------
    # FIRST: best.pt
    # --------------------------------------------------------

    results = best_model.predict(
        source=image,
        imgsz=640,
        conf=BEST_CONFIDENCE,
        verbose=False
    )

    detection = get_best_detection(
        results[0]
    )

    model_used = None

    if detection is not None:

        box, confidence = detection
        model_used = "best.pt"

    else:

        # ----------------------------------------------------
        # FALLBACK: last.pt
        # ----------------------------------------------------

        results = last_model.predict(
            source=image,
            imgsz=640,
            conf=LAST_CONFIDENCE,
            verbose=False
        )

        detection = get_best_detection(
            results[0]
        )

        if detection is None:

            print(
                f"[NO NOSE] "
                f"{os.path.basename(image_path)}"
            )

            # Save original image for review
            output_path = os.path.join(
                OUTPUT_DIR,
                os.path.basename(image_path)
            )

            cv2.imwrite(
                output_path,
                image
            )

            return

        box, confidence = detection
        model_used = "last.pt"

    # --------------------------------------------------------
    # BOX COORDINATES
    # --------------------------------------------------------

    x1, y1, x2, y2 = box.astype(int)

    x1 = max(
        0,
        min(x1, width - 1)
    )

    y1 = max(
        0,
        min(y1, height - 1)
    )

    x2 = max(
        0,
        min(x2, width)
    )

    y2 = max(
        0,
        min(y2, height)
    )

    if x2 <= x1 or y2 <= y1:

        print(
            f"[INVALID BOX] "
            f"{os.path.basename(image_path)}"
        )

        return

    # --------------------------------------------------------
    # DRAW RESULT
    # --------------------------------------------------------

    output = image.copy()

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        3
    )

    label = (
        f"{model_used} | "
        f"dog_nose {confidence:.2f}"
    )

    cv2.putText(
        output,
        label,
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        os.path.basename(image_path)
    )

    cv2.imwrite(
        output_path,
        output
    )

    print(
        f"[{model_used}] "
        f"{os.path.basename(image_path)} "
        f"-> confidence={confidence:.2f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("              NOSE DETECTOR FALLBACK")
    print("=" * 70)

    # --------------------------------------------------------
    # Check models
    # --------------------------------------------------------

    if not os.path.exists(BEST_MODEL):

        print()
        print("ERROR: best.pt not found:")
        print(BEST_MODEL)
        return

    if not os.path.exists(LAST_MODEL):

        print()
        print("ERROR: last.pt not found:")
        print(LAST_MODEL)
        return

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not os.path.exists(INPUT_DIR):

        print()
        print("ERROR: face_crops folder not found:")
        print(INPUT_DIR)
        return

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print()
    print("Loading best.pt...")
    best_model = YOLO(
        BEST_MODEL
    )

    print("Loading last.pt...")
    last_model = YOLO(
        LAST_MODEL
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
        f"Face crops found: "
        f"{len(image_files)}"
    )

    print()
    print("Detection strategy:")
    print("1. Try best.pt")
    print("2. If no detection, try last.pt")
    print("3. If both fail, report NO NOSE")
    print()

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    for image_name in sorted(
        image_files
    ):

        image_path = os.path.join(
            INPUT_DIR,
            image_name
        )

        process_image(
            image_path,
            best_model,
            last_model
        )

    print()
    print("=" * 70)
    print("              FALLBACK TEST COMPLETE")
    print("=" * 70)

    print()
    print("Results saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()