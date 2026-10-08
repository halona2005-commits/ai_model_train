import pandas as pd
from pathlib import Path
import json
import shutil
import cv2
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# ---------------- PATHS ----------------
ROOT = Path(r"D:\internship_RIED\ai-model")

IMAGE_ROOT = ROOT / "pet_biometric_clean" / "images"
CSV_PATH   = ROOT / "pet_biometric_clean" / "train_data_clean.csv"

DOGFLW_TRAIN_IMG = ROOT / "DogFLW" / "train" / "images"
DOGFLW_TRAIN_LBL = ROOT / "DogFLW" / "train" / "labels"

VAL_ROOT = ROOT / "dataset" / "validator"
RAW_ROOT = VAL_ROOT / "raw"
SPLITS   = VAL_ROOT / "splits"

DOG_NOSE_DIR = RAW_ROOT / "dog_nose"
NOT_NOSE_DIR = RAW_ROOT / "not_dog_nose"

EYE_DIR   = NOT_NOSE_DIR / "dog_eye"
MOUTH_DIR = NOT_NOSE_DIR / "dog_mouth"
FACE_DIR  = NOT_NOSE_DIR / "dog_face"

for d in [DOG_NOSE_DIR, EYE_DIR, MOUTH_DIR, FACE_DIR, SPLITS]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------- HELPERS ----------------
def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    return img

def find_image_for_label(base_name, img_dir):
    for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".bmp"]:
        c = img_dir / (base_name + ext)
        if c.exists():
            return c
    return None

def make_bbox(pts, W, H, pad=0.25, min_side=16):
    """
    pts: Nx2 float array (may contain NaN)
    returns (x1, y1, x2, y2) or None
    """
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return None

    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]
    if pts.shape[0] < 2:
        return None

    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    if not np.isfinite([x_min, y_min, x_max, y_max]).all():
        return None

    bw = float(x_max - x_min)
    bh = float(y_max - y_min)
    if bw < 2 or bh < 2:
        return None

    x1 = max(0, int(x_min - pad * bw))
    y1 = max(0, int(y_min - pad * bh))
    x2 = min(W, int(x_max + pad * bw))
    y2 = min(H, int(y_max + pad * bh))

    if (x2 - x1) < min_side or (y2 - y1) < min_side:
        return None

    return x1, y1, x2, y2

# ---------------- 0. SANITY ----------------
print("=" * 60)
print("SANITY: sample image sizes")
print("=" * 60)

for p in list(IMAGE_ROOT.glob("*"))[:3]:
    img = load_image(p)
    if img is not None:
        print(f"pet_biometric_clean/{p.name}: {img.shape[1]}x{img.shape[0]}")

for p in list(DOGFLW_TRAIN_IMG.glob("*"))[:3]:
    img = load_image(p)
    if img is not None:
        print(f"DogFLW/train/images/{p.name}: {img.shape[1]}x{img.shape[0]}")

# ---------------- 1. COPY NOSE POSITIVES ----------------
print()
print("=" * 60)
print("STEP 1: copy dog_nose positives")
print("=" * 60)

df = pd.read_csv(CSV_PATH)
print("Rows in train_data_clean.csv:", len(df))

copied, skipped = 0, 0
records = []

for _, row in df.iterrows():
    src = IMAGE_ROOT / str(row["nose print image"])
    if not src.exists():
        skipped += 1
        continue
    dst = DOG_NOSE_DIR / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
    copied += 1
    records.append({
        "image_path": f"dataset/validator/raw/dog_nose/{src.name}",
        "label": 1,
        "negative_type": "dog_nose",
        "group_id": f"dog_{row['dog ID']}",
    })

print("Copied dog_nose:", copied, "| skipped:", skipped)

# ---------------- 2. CROP NEGATIVES FROM DOGFLW ----------------
print()
print("=" * 60)
print("STEP 2: crop dog_eye / dog_mouth / dog_face from DogFLW")
print("=" * 60)

label_files = sorted(DOGFLW_TRAIN_LBL.glob("*.json"))
print("DogFLW train labels:", len(label_files))

eye_count, mouth_count, face_count, skip_count = 0, 0, 0, 0
eye_box_fail, mouth_box_fail, face_box_fail = 0, 0, 0

for lbl_path in label_files:
    base = lbl_path.stem

    img_path = find_image_for_label(base, DOGFLW_TRAIN_IMG)
    if img_path is None:
        skip_count += 1
        continue

    img = load_image(img_path)
    if img is None:
        skip_count += 1
        continue

    H, W = img.shape[:2]

    try:
        with open(lbl_path, "r") as f:
            data = json.load(f)
    except Exception:
        skip_count += 1
        continue

    lms_raw = data.get("landmarks")
    if lms_raw is None:
        skip_count += 1
        continue

    lms = np.array(lms_raw, dtype=np.float32)
    if lms.shape != (46, 2):
        skip_count += 1
        continue

    # Eye: landmarks 0-13
    eye_box = make_bbox(lms[0:14], W, H, pad=0.25)
    if eye_box is not None:
        x1, y1, x2, y2 = eye_box
        crop = img[y1:y2, x1:x2]
        if crop.size > 0 and crop.shape[0] >= 16 and crop.shape[1] >= 16:
            cv2.imwrite(str(EYE_DIR / f"{base}_eye.jpg"), crop)
            eye_count += 1
        else:
            eye_box_fail += 1
    else:
        eye_box_fail += 1

    # Mouth: landmarks 24-45
    mouth_box = make_bbox(lms[24:46], W, H, pad=0.20)
    if mouth_box is not None:
        x1, y1, x2, y2 = mouth_box
        crop = img[y1:y2, x1:x2]
        if crop.size > 0 and crop.shape[0] >= 16 and crop.shape[1] >= 16:
            cv2.imwrite(str(MOUTH_DIR / f"{base}_mouth.jpg"), crop)
            mouth_count += 1
        else:
            mouth_box_fail += 1
    else:
        mouth_box_fail += 1

    # Face: bounding_boxes
    bb = data.get("bounding_boxes")
    if bb and len(bb) == 4:
        try:
            x1 = int(float(bb[0]))
            y1 = int(float(bb[1]))
            x2 = int(float(bb[2]))
            y2 = int(float(bb[3]))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if (x2 - x1) >= 32 and (y2 - y1) >= 32:
                crop = img[y1:y2, x1:x2]
                cv2.imwrite(str(FACE_DIR / f"{base}_face.jpg"), crop)
                face_count += 1
            else:
                face_box_fail += 1
        except Exception:
            face_box_fail += 1
    else:
        face_box_fail += 1

print("Eye crops:  ", eye_count, " (failed:", eye_box_fail, ")")
print("Mouth crops:", mouth_count, " (failed:", mouth_box_fail, ")")
print("Face crops: ", face_count, " (failed:", face_box_fail, ")")
print("Skipped:    ", skip_count)

# ---------------- 3. BUILD NEGATIVE RECORDS ----------------
def breed_of(p: Path):
    return p.stem.split("_")[0]

for subdir, neg_type in [(EYE_DIR, "dog_eye"), (MOUTH_DIR, "dog_mouth"), (FACE_DIR, "dog_face")]:
    for p in subdir.glob("*.jpg"):
        records.append({
            "image_path": f"dataset/validator/raw/not_dog_nose/{neg_type}/{p.name}",
            "label": 0,
            "negative_type": neg_type,
            "group_id": f"breed_{breed_of(p)}",
        })

all_df = pd.DataFrame(records)
print()
print("=" * 60)
print("COMBINED DATASET")
print("=" * 60)
print("Total:", len(all_df))
print(all_df["negative_type"].value_counts().to_string())

# ---------------- 4. GROUPED SPLIT ----------------
print()
print("=" * 60)
print("SPLITTING (grouped by dog ID / breed)")
print("=" * 60)

gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
tr_idx, tmp_idx = next(gss1.split(all_df, groups=all_df["group_id"]))

train_df = all_df.iloc[tr_idx].copy().reset_index(drop=True)
tmp_df   = all_df.iloc[tmp_idx].copy().reset_index(drop=True)

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
va_idx, te_idx = next(gss2.split(tmp_df, groups=tmp_df["group_id"]))

val_df  = tmp_df.iloc[va_idx].copy().reset_index(drop=True)
test_df = tmp_df.iloc[te_idx].copy().reset_index(drop=True)

train_df.to_csv(SPLITS / "train.csv", index=False)
val_df.to_csv(SPLITS / "val.csv", index=False)
test_df.to_csv(SPLITS / "test.csv", index=False)

print()
for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
    pos = int((d["label"] == 1).sum())
    neg = int((d["label"] == 0).sum())
    print(f"{name}: total={len(d)}  pos={pos}  neg={neg}")
    print(d["negative_type"].value_counts().to_string())
    print()

print("CSVs saved:")
print(" ", SPLITS / "train.csv")
print(" ", SPLITS / "val.csv")
print(" ", SPLITS / "test.csv")