import sys
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.applications.resnet50 import preprocess_input
import json

# Paths
ROOT = Path(r"D:\internship_RIED\ai-model")
sys.path.insert(0, str(ROOT))

from src.pipeline.prepare_nose import prepare_nose
from src.pipeline.recognize import _get_model # Reuse the loaded model

DATA_ROOT = ROOT / "pet_biometric_full"
SPLITS = DATA_ROOT / "splits"
DB = ROOT / "dataset" / "biometric" / "database"
DB.mkdir(parents=True, exist_ok=True)

print("Loading Train CSV...")
train_df = pd.read_csv(SPLITS / "train.csv")
# Limit to 2000 images for speed, or use all 13k if you have time
# train_df = train_df.head(2000) 

model = _get_model()

def embed_crop(crop_bgr):
    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32)
    img = preprocess_input(img)
    vec = model.predict(np.expand_dims(img, 0), verbose=0)[0]
    return vec

print("Building Database with YOLO-aligned crops...")
valid_embs = []
valid_ids = []
valid_paths = []

for i, row in train_df.iterrows():
    img_path = DATA_ROOT / "train" / "images" / row["image"]
    
    # Use the EXACT same function as recognize()
    prep = prepare_nose(image_path=img_path)
    
    if prep["status"] == "accepted":
        vec = embed_crop(prep["crop_bgr"])
        valid_embs.append(vec)
        valid_ids.append(row["dog_id"])
        valid_paths.append(str(img_path))
        
    if (i+1) % 100 == 0:
        print(f"Processed {i+1}/{len(train_df)} | Accepted: {len(valid_embs)}")

print("Saving to database...")
emb_array = np.stack(valid_embs).astype(np.float32)
ids_array = np.array(valid_ids, dtype=np.int32)
paths_array = np.array(valid_paths)

np.save(DB / "embeddings.npy", emb_array)
np.save(DB / "dog_ids.npy", ids_array)
np.save(DB / "paths.npy", paths_array)

unique_dogs = sorted(int(d) for d in np.unique(ids_array))
metadata = {
    "total_vectors": int(len(emb_array)),
    "total_dogs": len(unique_dogs),
    "dog_ids": unique_dogs,
    "embedding_dim": int(emb_array.shape[1]),
}
with open(DB / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Database rebuilt with {len(emb_array)} YOLO-aligned crops!")