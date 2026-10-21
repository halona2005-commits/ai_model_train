"""
Visual test that augmentation actually rotates.

Loads one nose image, applies augmentation 8 times,
saves a montage so you can see the variation.

Expected: 8 images that look like the same nose
          but rotated, brightened, zoomed.
"""

from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

from src.biometric.augmentation import augment_train

ROOT = Path(r"D:\internship_RIED\ai-model")
SPLITS = ROOT / "dataset_corrected" / "train"
MANIFEST = ROOT / "dataset_corrected" / "manifest.csv"
OUT = ROOT / "dataset_corrected" / "augmentation_test"
OUT.mkdir(parents=True, exist_ok=True)

import pandas as pd

df = pd.read_csv(MANIFEST)
df = df[(df["split"] == "train") & (df["keep"] == 1)]

# pick a dog with at least 3 clusters → most variation to show
dog_clusters = df.groupby("dog_id")["cluster_id"].nunique()
multi = dog_clusters[dog_clusters >= 3]
if len(multi) == 0:
    multi = dog_clusters[dog_clusters >= 2]

example_dog = multi.index[0]
sub = df[df["dog_id"] == example_dog]
image_name = sub.iloc[0]["image"]

src = SPLITS / "images" / image_name
print(f"Example dog : {example_dog}")
print(f"Image       : {image_name}")
print(f"Path        : {src}")

img_bgr = cv2.imread(str(src))
if img_bgr is None:
    raise RuntimeError(f"Cannot read {src}")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (224, 224))

# --- original ---
original = img_resized.copy()

# --- 8 augmented versions ---
augmented = []
for i in range(8):
    tensor = tf.constant(img_resized, dtype=tf.float32)
    out = augment_train(tensor).numpy()
    out = np.clip(out, 0, 255).astype(np.uint8)
    augmented.append(out)

# --- montage: 3 rows x 3 cols = original + 8 augmented ---
tiles = [original] + augmented
tiles = [cv2.resize(t, (224, 224)) for t in tiles]

row1 = np.concatenate(tiles[0:3], axis=1)
row2 = np.concatenate(tiles[3:6], axis=1)
row3 = np.concatenate(tiles[6:9], axis=1)
montage = np.concatenate([row1, row2, row3], axis=0)

# add labels
label = "original    | aug1    | aug2 | aug3    | aug4 | aug5    | aug6 | aug7    | aug8"
cv2.putText(
    montage, "original          augmentations ->",
    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
)
# yellow borders around augmented (not original)
for i in range(1, 9):
    r, c = (i // 3), (i % 3)
    y1 = r * 224
    x1 = c * 224
    cv2.rectangle(montage, (x1 + 1, y1 + 1), (x1 + 222, y1 + 222),
                  (0, 255, 0), 2)

out_path = OUT / f"aug_test_dog{example_dog}.jpg"
cv2.imwrite(str(out_path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))

print()
print(f"Saved montage: {out_path}")
print()
print("Open this file. You should see:")
print("  Top-left:   original")
print("  Other 8:    rotated, brightened, zoomed versions")
print()
print("If all 9 look identical -> augmentation still broken.")
print("If they differ -> augmentation works.")