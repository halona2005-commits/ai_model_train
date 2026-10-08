from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

ROOT   = Path(r"D:\internship_RIED\ai-model")
SPLITS = ROOT / "dataset" / "validator" / "splits"
MODEL  = ROOT / "models" / "nose_validator" / "best_phase1.keras"

IMG_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE

print("Loading model:", MODEL)
model = tf.keras.models.load_model(MODEL)

test_df = pd.read_csv(SPLITS / "test.csv")
test_df["abs_path"] = test_df["image_path"].apply(lambda p: str(ROOT / p))

def decode_img(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img

paths  = test_df["abs_path"].values
labels = test_df["label"].values
types  = test_df["negative_type"].values

ds = tf.data.Dataset.from_tensor_slices(paths)
ds = ds.map(decode_img, num_parallel_calls=AUTOTUNE)
ds = ds.batch(32).prefetch(AUTOTUNE)

preds = model.predict(ds, verbose=1).ravel()

print("\n" + "=" * 60)
print("PER-TYPE ANALYSIS")
print("=" * 60)
print(f"{'type':<12} {'n':>6} {'mean_p':>9} {'p5':>8} {'p95':>8}")
print("-" * 50)
for t in sorted(set(types)):
    mask = types == t
    p = preds[mask]
    print(f"{t:<12} {len(p):>6} {p.mean():>9.4f} "
          f"{np.percentile(p,5):>8.4f} {np.percentile(p,95):>8.4f}")

print("\n" + "=" * 60)
print("CRITICAL CHECKS")
print("=" * 60)

nose = preds[types == "dog_nose"]
print(f"dog_nose with p>=0.5: {100*(nose>=0.5).mean():.2f}%  (want HIGH)")

for t in ["dog_eye", "dog_mouth", "dog_face"]:
    p = preds[types == t]
    print(f"{t} with p<0.5:  {100*(p<0.5).mean():.2f}%  (want HIGH)")

acc = ((preds >= 0.5).astype(int) == labels).mean()
print(f"\nOverall test accuracy: {acc:.4f}")