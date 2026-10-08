from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"D:\internship_RIED\ai-model")
CSV  = ROOT / "pet_biometric_clean" / "train_data_clean.csv"
IMG  = ROOT / "pet_biometric_clean" / "images"

df = pd.read_csv(CSV)
df.columns = ["dog_id", "image"]

print("=" * 60)
print("RAW DATA")
print("=" * 60)
print("Total rows:", len(df))
print("Unique dogs:", df["dog_id"].nunique())

# check all files exist
missing = 0
for fn in df["image"]:
    if not (IMG / fn).exists():
        missing += 1
print("Missing image files:", missing)

counts = df["dog_id"].value_counts()

print()
print("=" * 60)
print("IMAGES PER DOG")
print("=" * 60)
print("min :", counts.min())
print("max :", counts.max())
print("mean:", round(counts.mean(), 2))
print("median:", counts.median())

print()
print("=" * 60)
print("DISTRIBUTION")
print("=" * 60)

bins = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 100), (101, 1000)]
for lo, hi in bins:
    n_dogs = ((counts >= lo) & (counts <= hi)).sum()
    n_imgs = counts[(counts >= lo) & (counts <= hi)].sum()
    label = f"{lo}" if lo == hi else f"{lo}-{hi}"
    print(f"  {label:>8} images per dog: {n_dogs:>4} dogs  ({n_imgs} images)")

print()
print("=" * 60)
print("USABILITY FOR TRIPLET TRAINING")
print("=" * 60)

usable = counts[counts >= 2]
unusable = counts[counts == 1]

print(f"Dogs with >=2 images (usable for anchor-positive): {len(usable)}")
print(f"Dogs with exactly 1 image (skip or use augmentation): {len(unusable)}")
print()

total_usable_images = usable.sum()
print(f"Total usable images: {total_usable_images} / {len(df)}")

print()
print("=" * 60)
print("TRIPLET POTENTIAL")
print("=" * 60)
# For each dog with k images, number of unique anchor-positive pairs = C(k,2)
total_pairs = sum(k * (k - 1) // 2 for k in counts[counts >= 2])
print(f"Total possible anchor-positive pairs: {total_pairs}")

# Same for full triplet combos: AP pairs * negatives from other dogs
# Rough estimate: pairs_per_dog * total_images_of_other_dogs
print("Estimated triplet combinations available: millions (more than enough)")