from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import collections

from src.biometric.batch_sampler import make_pk_dataset

ROOT = Path(r"D:\internship_RIED\ai-model")
SPLITS = ROOT / "dataset" / "biometric" / "splits"
OUT = ROOT / "dataset" / "biometric" / "batch_preview"
OUT.mkdir(parents=True, exist_ok=True)

# ---- load train split ----
train_df = pd.read_csv(SPLITS / "train.csv")
print("Train rows:", len(train_df))
print("Train dogs:", train_df["dog_id"].nunique())

# ---- global dog index mapping ----
all_dogs = sorted(train_df["dog_id"].unique().tolist())
dog_to_idx = {d: i for i, d in enumerate(all_dogs)}
print("Global number of train classes:", len(dog_to_idx))

# ---- make dataset ----
ds, gen = make_pk_dataset(
    train_df,
    dog_to_idx,
    P=8,
    K=4,
    augment=True,
    batches_per_epoch=5,
    seed=42,
)

# ---- fetch one batch ----
print("\nFetching one batch...")
for imgs, labels in ds.take(1):
    imgs_np = imgs.numpy()
    labels_np = labels.numpy()

    print("Batch images shape:", imgs_np.shape)
    print("Batch labels      :", labels_np)
    print("Label counts      :", dict(collections.Counter(labels_np.tolist())))

    # ---- reverse ResNet50 preprocessing so we can see them ----
    # preprocess_input subtracts channel means. Add them back.
    # channel order for preprocess_input in Keras (TF): RGB but with BGR means
    # the Keras resnet50.preprocess_input on TF does: x[..., ::-1] then subtract,
    # then reverse. Safest is to undo by adding back the means directly.
    MEANS = np.array([123.68, 116.779, 103.939], dtype=np.float32)  # R,G,B

    vis = imgs_np + MEANS  # broadcast over channels
    vis = np.clip(vis, 0, 255).astype(np.uint8)

    # ---- montage: 8 rows (dogs) x 4 cols (images) ----
    rows = []
    for p in range(8):
        row_imgs = vis[p * 4:(p + 1) * 4]
        row = np.concatenate([cv2.resize(im, (224, 224)) for im in row_imgs], axis=1)
        rows.append(row)
    montage = np.concatenate(rows, axis=0)

    out_path = OUT / "batch_preview.jpg"
    cv2.imwrite(str(out_path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    print("Saved:", out_path)

print("\nDone.")