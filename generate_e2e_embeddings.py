import os
import zipfile
import shutil
from pathlib import Path
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import preprocess_input

# Paths
PROJECT_ROOT = Path(r"D:\internship_RIED\ai-model")
DATA_ROOT = PROJECT_ROOT / "pet_biometric_full"
SPLITS = DATA_ROOT / "splits"
IMG_ROOT = DATA_ROOT / "train" / "images"
MODEL_PATH = PROJECT_ROOT / "models" / "biometric" / "best_biometric_arcface.keras"
OUT_DIR = PROJECT_ROOT / "dataset" / "biometric" / "embeddings_v3"
EXTRACT_DIR = PROJECT_ROOT / "temp_model"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Model (Bypassing Keras bug)
if EXTRACT_DIR.exists(): shutil.rmtree(EXTRACT_DIR)
EXTRACT_DIR.mkdir()
with zipfile.ZipFile(MODEL_PATH, 'r') as z: z.extractall(EXTRACT_DIR)

class ArcFaceLayer(layers.Layer):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def build(self, input_shape): self.W = self.add_weight(shape=(input_shape[0][-1], 4200))
    def call(self, inputs): return inputs[0]

class L2Norm(layers.Layer):
    def call(self, x): return tf.math.l2_normalize(x, axis=1)

backbone = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=(224, 224, 3), pooling="avg")
inp = tf.keras.Input(shape=(224, 224, 3))
x = L2Norm()(backbone(inp))
model = tf.keras.Model(inputs=inp, outputs=x)
model.load_weights(EXTRACT_DIR / "model.weights.h5")
shutil.rmtree(EXTRACT_DIR)

# 2. Embed function
def load_and_embed(df):
    paths = [str(IMG_ROOT / img) for img in df["image"]]
    def decode_img(path):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (224, 224))
        img = tf.cast(img, tf.float32)
        return preprocess_input(img)
    
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(decode_img, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE)
    return model.predict(ds, verbose=1)

# 3. Process sets
print("Generating Train embeddings...")
train_df = pd.read_csv(SPLITS / "train.csv")
train_emb = load_and_embed(train_df)
np.save(OUT_DIR / "train_embeddings.npy", train_emb)
np.save(OUT_DIR / "train_labels.npy", train_df["dog_id"].values)
np.save(OUT_DIR / "train_paths.npy", train_df["abs_path"].values)

print("Generating Test embeddings...")
test_df = pd.read_csv(SPLITS / "test.csv")
test_emb = load_and_embed(test_df)
np.save(OUT_DIR / "test_embeddings.npy", test_emb)
np.save(OUT_DIR / "test_labels.npy", test_df["dog_id"].values)
np.save(OUT_DIR / "test_paths.npy", test_df["abs_path"].values)