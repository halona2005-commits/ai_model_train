"""
Generate embeddings using the v2 biometric model.

Reads:
    dataset_corrected/splits/{train,val,test}.csv

Writes:
    dataset_corrected/embeddings_v2/{train,val,test}_embeddings.npy
    dataset_corrected/embeddings_v2/{train,val,test}_labels.npy
    dataset_corrected/embeddings_v2/{train,val,test}_clusters.npy
    dataset_corrected/embeddings_v2/{train,val,test}_paths.npy
"""

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

ROOT = Path(r"D:\internship_RIED\ai-model")
SPLITS = ROOT / "dataset_corrected" / "splits"
MODEL = ROOT / "models" / "biometric" / "best_biometric_v2.keras"
OUT = ROOT / "dataset_corrected" / "embeddings_v2"
OUT.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH = 32
AUTOTUNE = tf.data.AUTOTUNE

print("Loading model:", MODEL)
model = tf.keras.models.load_model(MODEL, compile=False)
print("Output shape:", model.output_shape)


def decode_img(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img


def embed_split(name):
    df = pd.read_csv(SPLITS / f"{name}.csv")
    paths = df["abs_path"].values
    print(f"\n[{name}] embedding {len(paths)} images...")

    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(decode_img, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(AUTOTUNE)

    embs = model.predict(ds, verbose=1).astype(np.float32)

    np.save(OUT / f"{name}_embeddings.npy", embs)
    np.save(OUT / f"{name}_labels.npy", df["dog_id"].values)
    np.save(OUT / f"{name}_clusters.npy", df["cluster_id"].values)
    np.save(OUT / f"{name}_paths.npy", df["abs_path"].values)

    print(f"[{name}] saved {embs.shape}")


for split in ["train", "val", "test"]:
    embed_split(split)

print()
print("Done. Outputs in:", OUT)
for f in sorted(OUT.glob("*.npy")):
    print(f"  {f.name}  {f.stat().st_size/1e6:.2f} MB")