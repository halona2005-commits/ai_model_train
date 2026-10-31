from pathlib import Path
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2

from src.pipeline.prepare_nose import prepare_nose
# We import the load model and embed function from recognize.py to keep it consistent
from src.pipeline.recognize import _get_model, _embed_crop

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "dataset" / "biometric" / "database"
MATCH_THRESHOLD = 0.80  # If similarity is above this, it's a duplicate

def register_dog(nose_image_path, dog_name, dog_id=None):
    """
    Register a dog. Refuses if prepare_nose rejects the image.
    Refuses if the dog is already in the database (similarity > 0.80).
    """
    # 1. Validate the nose
    prep = prepare_nose(image_path=nose_image_path)
    if prep["status"] != "accepted":
        return {
            "status": "rejected",
            "reason_code": prep["reason_code"],
            "message": f"Cannot register: {prep['reason_code']}",
        }

    # 2. Generate Embedding
    vec = _embed_crop(prep["crop_bgr"])

    # 3. Load Database
    emb = np.load(DB / "embeddings.npy")
    ids = np.load(DB / "dog_ids.npy")
    paths = np.load(DB / "paths.npy", allow_pickle=True)
    with open(DB / "metadata.json") as f:
        meta = json.load(f)

    # 4. CHECK FOR DUPLICATES (The Fix)
    if len(emb) > 0:
        sims = emb @ vec
        max_sim = float(np.max(sims))
        
        if max_sim >= MATCH_THRESHOLD:
            best_match_idx = int(np.argmax(sims))
            existing_dog_id = int(ids[best_match_idx])
            return {
                "status": "already_registered",
                "dog_id": existing_dog_id,
                "similarity": max_sim,
                "message": f"This dog is already registered as Dog ID {existing_dog_id} (Similarity: {max_sim:.2f})"
            }

    # 5. Assign new ID and Save
    if dog_id is None:
        dog_id = max(meta["dog_ids"]) + 1 if meta["dog_ids"] else 0
    dog_id = int(dog_id)

    new_emb = np.concatenate([emb, vec[None, :]], axis=0).astype(np.float32)
    new_ids = np.concatenate([ids, [dog_id]]).astype(np.int32)
    new_paths = np.concatenate([paths, [str(nose_image_path)]])

    np.save(DB / "embeddings.npy", new_emb)
    np.save(DB / "dog_ids.npy", new_ids)
    np.save(DB / "paths.npy", new_paths)

    if dog_id not in meta["dog_ids"]:
        meta["dog_ids"].append(dog_id)
        meta["dog_ids"] = sorted(set(meta["dog_ids"]))
    meta["total_vectors"] = int(len(new_emb))
    meta["total_dogs"] = len(meta["dog_ids"])
    with open(DB / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "status": "registered",
        "dog_id": dog_id,
        "dog_name": dog_name,
        "message": "Successfully registered new dog."
    }