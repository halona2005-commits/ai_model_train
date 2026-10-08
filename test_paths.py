import pandas as pd
from pathlib import Path

ROOT = Path(r"D:\internship_RIED\ai-model")
IMAGE_ROOT = ROOT / "pet_biometric_clean" / "images"

print("=" * 60)
print("IMAGE ROOT CHECK")
print("=" * 60)
print("Image root exists:", IMAGE_ROOT.exists())

if IMAGE_ROOT.exists():
    imgs = list(IMAGE_ROOT.glob("*"))
    print("Total files in images folder:", len(imgs))
    if imgs:
        print("Sample files:")
        for p in imgs[:5]:
            print("  ", p.name)

print()
print("=" * 60)
print("CSV CHECK")
print("=" * 60)

for name in ["train_data_clean.csv", "image_quality_scores.csv"]:
    p = ROOT / "pet_biometric_clean" / name
    print(f"{name}: exists={p.exists()}  path={p}")

print()
print("=" * 60)
print("LINKAGE CHECK (CSV filename -> image file)")
print("=" * 60)

csv_path = ROOT / "pet_biometric_clean" / "train_data_clean.csv"
df = pd.read_csv(csv_path)
print("Rows in train_data_clean.csv:", len(df))
print("Columns:", df.columns.tolist())

sample = df.head(5)
ok = 0
missing = 0
for _, row in sample.iterrows():
    fn = row["nose print image"]
    full = IMAGE_ROOT / fn
    exists = full.exists()
    if exists:
        ok += 1
    else:
        missing += 1
    print(f"  {fn}  ->  exists={exists}")

print()
print(f"Sample check: found={ok}, missing={missing}")

# Full check
all_ok = 0
all_missing = 0
missing_examples = []
for fn in df["nose print image"].astype(str):
    if (IMAGE_ROOT / fn).exists():
        all_ok += 1
    else:
        all_missing += 1
        if len(missing_examples) < 5:
            missing_examples.append(fn)

print()
print("=" * 60)
print("FULL CHECK")
print("=" * 60)
print("Total rows:", len(df))
print("Found in images folder:", all_ok)
print("Missing:", all_missing)
if missing_examples:
    print("First missing examples:")
    for m in missing_examples:
        print("  ", m)

print()
print("=" * 60)
print("DOGFLW CHECK")
print("=" * 60)
for sub in ["train/images", "train/labels", "test/images", "test/labels"]:
    p = ROOT / "DogFLW" / sub
    n = len(list(p.glob("*"))) if p.exists() else -1
    print(f"DogFLW/{sub}: exists={p.exists()}  files={n}")