import os
import cv2
import numpy as np

from pathlib import Path

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    r"E:\canine-nose-print-recognition\ai-model"
)

INPUT_DIR = (
    BASE_DIR /
    "nose_v3_batch" /
    "enhanced"
)

OUTPUT_DIR = (
    BASE_DIR /
    "features" /
    "runtime"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# RESNET50
# ============================================================

print()
print("=" * 70)
print("          RESNET50 RUNTIME FEATURE EXTRACTION")
print("=" * 70)

print()
print("Loading ResNet50...")

model = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

print("ResNet50 loaded.")


# ============================================================
# SUPPORTED IMAGES
# ============================================================

EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


image_files = sorted([
    p
    for p in INPUT_DIR.iterdir()
    if p.suffix.lower() in EXTENSIONS
])


print()
print(f"Input folder : {INPUT_DIR}")
print(f"Images found : {len(image_files)}")
print()


# ============================================================
# FEATURE EXTRACTION
# ============================================================

features = []
image_names = []

failed = 0


for index, image_path in enumerate(
    image_files,
    start=1
):

    print(
        f"[{index}/{len(image_files)}] "
        f"{image_path.name}"
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            "   FAILED: could not read image"
        )

        failed += 1
        continue


    try:

        # ----------------------------------------------------
        # BGR → RGB
        # ----------------------------------------------------

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # Resize to ResNet50 input size
        # ----------------------------------------------------

        image = cv2.resize(
            image,
            (224, 224),
            interpolation=cv2.INTER_AREA
        )


        # ----------------------------------------------------
        # Float32
        # ----------------------------------------------------

        image = image.astype(
            np.float32
        )


        # ----------------------------------------------------
        # ResNet50 preprocessing
        # ----------------------------------------------------

        image = preprocess_input(
            image
        )


        # ----------------------------------------------------
        # Add batch dimension
        # ----------------------------------------------------

        image = np.expand_dims(
            image,
            axis=0
        )


        # ----------------------------------------------------
        # Extract 2048-D feature vector
        # ----------------------------------------------------

        feature = model.predict(
            image,
            verbose=0
        )


        # ----------------------------------------------------
        # Verify shape
        # ----------------------------------------------------

        if feature.shape != (1, 2048):

            print(
                f"   FAILED: unexpected shape "
                f"{feature.shape}"
            )

            failed += 1
            continue


        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        features.append(
            feature[0]
        )

        image_names.append(
            image_path.name
        )


        print(
            "   OK → 2048-D"
        )


    except Exception as e:

        print(
            f"   FAILED: {e}"
        )

        failed += 1


# ============================================================
# CONVERT TO NUMPY ARRAYS
# ============================================================

if len(features) == 0:

    raise RuntimeError(
        "No features were extracted."
    )


features = np.array(
    features,
    dtype=np.float32
)

image_names = np.array(
    image_names
)


# ============================================================
# SAVE
# ============================================================

FEATURE_PATH = (
    OUTPUT_DIR /
    "runtime_features.npy"
)

NAMES_PATH = (
    OUTPUT_DIR /
    "runtime_image_names.npy"
)


np.save(
    FEATURE_PATH,
    features
)

np.save(
    NAMES_PATH,
    image_names
)


# ============================================================
# FINAL VERIFICATION
# ============================================================

print()
print("=" * 70)
print("          FEATURE EXTRACTION COMPLETE")
print("=" * 70)

print()
print(
    f"Images processed : {len(image_names)}"
)

print(
    f"Failed           : {failed}"
)

print(
    f"Feature shape    : {features.shape}"
)

print(
    f"Feature dtype    : {features.dtype}"
)

print()
print(
    f"Features saved to:"
)

print(
    FEATURE_PATH
)

print()
print(
    f"Image names saved to:"
)

print(
    NAMES_PATH
)

print()
print("=" * 70)