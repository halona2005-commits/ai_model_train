import numpy as np
from pathlib import Path


BASE_DIR = Path(
    r"E:\canine-nose-print-recognition\ai-model"
)

FEATURE_PATH = (
    BASE_DIR /
    "features" /
    "runtime" /
    "runtime_features.npy"
)

NAMES_PATH = (
    BASE_DIR /
    "features" /
    "runtime" /
    "runtime_image_names.npy"
)


print()
print("=" * 70)
print("          VERIFY RUNTIME FEATURES")
print("=" * 70)


# ============================================================
# LOAD
# ============================================================

features = np.load(
    FEATURE_PATH
)

image_names = np.load(
    NAMES_PATH
)


print()
print("Feature file loaded.")
print("Names file loaded.")


# ============================================================
# BASIC CHECKS
# ============================================================

print()
print(
    f"Feature shape : {features.shape}"
)

print(
    f"Feature dtype : {features.dtype}"
)

print(
    f"Image count   : {len(image_names)}"
)


# ============================================================
# SHAPE CHECK
# ============================================================

shape_ok = (
    features.ndim == 2
    and features.shape[1] == 2048
)


# ============================================================
# COUNT CHECK
# ============================================================

count_ok = (
    len(features) ==
    len(image_names)
)


# ============================================================
# NAN / INF CHECK
# ============================================================

nan_count = np.isnan(
    features
).sum()

inf_count = np.isinf(
    features
).sum()

values_ok = (
    nan_count == 0
    and inf_count == 0
)


# ============================================================
# DISPLAY FIRST FEW
# ============================================================

print()
print("First 5 image mappings:")

for i in range(
    min(5, len(image_names))
):

    print(
        f"{i + 1}. "
        f"{image_names[i]} "
        f"→ {features[i].shape}"
    )


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 70)

if (
    shape_ok
    and count_ok
    and values_ok
):

    print(
        "FEATURE VERIFICATION: PASSED"
    )

else:

    print(
        "FEATURE VERIFICATION: FAILED"
    )


print("=" * 70)

print()
print(
    f"Shape check : {'PASS' if shape_ok else 'FAIL'}"
)

print(
    f"Count check : {'PASS' if count_ok else 'FAIL'}"
)

print(
    f"NaN values  : {nan_count}"
)

print(
    f"Inf values  : {inf_count}"
)

print("=" * 70)