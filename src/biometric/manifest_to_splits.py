"""
Convert manifest.csv into train/val/test CSVs that batch_sampler can read.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(r"D:\internship_RIED\ai-model")
MANIFEST = ROOT / "dataset_corrected" / "manifest.csv"
IMG_ROOT = ROOT / "dataset_corrected" / "train" / "images"
OUT = ROOT / "dataset_corrected" / "splits"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST)
print(f"Manifest rows : {len(df)}")

df = df[df["keep"] == 1].copy()
print(f"Kept rows     : {len(df)}")

df["abs_path"] = df["image"].apply(lambda x: str(IMG_ROOT / x))

missing = (~df["abs_path"].apply(lambda p: Path(p).exists())).sum()
print(f"Missing files : {missing}")
assert missing == 0

for split in ["train", "val", "test"]:
    sub = df[df["split"] == split][["image", "dog_id", "cluster_id", "abs_path"]]
    sub = sub.reset_index(drop=True)
    out_path = OUT / f"{split}.csv"
    sub.to_csv(out_path, index=False)

    n_dogs = sub["dog_id"].nunique()
    n_imgs = len(sub)
    clusters_per_dog = sub.groupby("dog_id")["cluster_id"].nunique()
    multi = (clusters_per_dog >= 2).sum()
    print(f"[{split}] dogs={n_dogs}  images={n_imgs}  "
          f"dogs_with_>=2_clusters={multi}  -> {out_path.name}")

print()
print("Done. Split CSVs in:", OUT)