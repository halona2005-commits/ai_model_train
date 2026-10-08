from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# ---------------- CONFIG ----------------
ROOT = Path(r"D:\internship_RIED\ai-model")
SPLITS = ROOT / "dataset" / "biometric" / "splits"
MODEL = ROOT / "models" / "biometric" / "best_biometric.keras"
OUT = ROOT / "dataset" / "biometric" / "embeddings"
OUT.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH = 32
AUTOTUNE = tf.data.AUTOTUNE

print("Loading model:", MODEL)
model = tf.keras.models.load_model(MODEL, compile=False)
print("Model output shape:", model.output_shape)


def decode_img(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img


def embed_images(df, name):
    paths = df["abs_path"].values
    print(f"\n[{name}] embedding {len(paths)} images...")

    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(decode_img, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(AUTOTUNE)

    embs = model.predict(ds, verbose=1)     # [N, 2048]
    embs = embs.astype(np.float32)

    np.save(OUT / f"{name}_embeddings.npy", embs)
    np.save(OUT / f"{name}_labels.npy", df["dog_id"].values)
    np.save(OUT / f"{name}_paths.npy", df["abs_path"].values)

    print(f"[{name}] saved {embs.shape} embeddings")
    return embs


# ---------------- TEST ----------------
test_df = pd.read_csv(SPLITS / "test.csv")
embed_images(test_df, "test")

# ---------------- ENROLLMENT ----------------
enroll_df = pd.read_csv(SPLITS / "enrollment.csv")
embed_images(enroll_df, "enrollment")

# ---------------- TRAIN (optional, useful for gallery) ----------------
train_df = pd.read_csv(SPLITS / "train.csv")
embed_images(train_df, "train")

print("\nDone. Files in:", OUT)
for f in sorted(OUT.glob("*.npy")):
    print(f"  {f.name:<28} {f.stat().st_size/1e6:.2f} MB")