from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"D:\internship_RIED\ai-model")
CSV  = ROOT / "pet_biometric_clean" / "train_data_clean.csv"
IMG  = ROOT / "pet_biometric_clean" / "images"
OUT  = ROOT / "dataset" / "biometric" / "splits"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42

# ---- load ----
df = pd.read_csv(CSV)
df.columns = ["dog_id", "image"]
df["abs_path"] = df["image"].apply(lambda x: str(IMG / x))

# verify all files exist
missing = (~df["abs_path"].apply(lambda p: Path(p).exists())).sum()
print("Missing files:", missing)
assert missing == 0

# ---- categorize dogs ----
counts = df["dog_id"].value_counts()
multi_dogs  = counts[counts >= 2].index.tolist()   # 960
single_dogs = counts[counts == 1].index.tolist()   # 104

print(f"Dogs with >=2 images: {len(multi_dogs)}")
print(f"Dogs with exactly 1:  {len(single_dogs)}")

# ---- shuffle dog IDs (deterministic) ----
rng = np.random.default_rng(SEED)
rng.shuffle(multi_dogs)

n = len(multi_dogs)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)

train_dogs = set(multi_dogs[:n_train])
val_dogs   = set(multi_dogs[n_train:n_train + n_val])
test_dogs  = set(multi_dogs[n_train + n_val:])

# ---- save ----
def save_split(dog_set, name):
    sub = df[df["dog_id"].isin(dog_set)].copy()
    sub = sub[["image", "dog_id", "abs_path"]].reset_index(drop=True)
    sub.to_csv(OUT / f"{name}.csv", index=False)
    n_imgs = len(sub)
    n_dogs = sub["dog_id"].nunique()
    imgs_per_dog = sub.groupby("dog_id").size()
    print(f"{name:<12}  dogs={n_dogs:<5}  images={n_imgs:<5}  "
          f"imgs/dog min={imgs_per_dog.min()} max={imgs_per_dog.max()}")

print()
print("=" * 60)
print("SPLITS")
print("=" * 60)
save_split(train_dogs, "train")
save_split(val_dogs, "val")
save_split(test_dogs, "test")

# enrollment-only (single-image dogs)
enroll = df[df["dog_id"].isin(single_dogs)].copy()
enroll = enroll[["image", "dog_id", "abs_path"]].reset_index(drop=True)
enroll.to_csv(OUT / "enrollment.csv", index=False)
print(f"{'enrollment':<12}  dogs={len(single_dogs):<5}  images={len(enroll):<5}  "
      "(single-image, test-only)")

print()
print("CSVs saved to:", OUT)
for f in sorted(OUT.glob("*.csv")):
    print("  ", f.name)

# sanity: no dog appears in more than one split
assert len(train_dogs & val_dogs) == 0
assert len(train_dogs & test_dogs) == 0
assert len(val_dogs & test_dogs) == 0
assert len((train_dogs | val_dogs | test_dogs) & set(single_dogs)) == 0
print()
print("Sanity check: no dog is shared across splits ✅")