from pathlib import Path
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

ROOT = Path(r"D:\internship_RIED\ai-model")
DB = ROOT / "dataset" / "biometric" / "database"
MODEL_BIOMETRIC = ROOT / "models" / "biometric" / "best_biometric.keras"

IMG_SIZE = (224, 224)

_biometric = None

def _get_model():
    global _biometric
    if _biometric is None:
        _biometric = tf.keras.models.load_model(MODEL_BIOMETRIC, compile=False)
    return _biometric

def _embed(path):
    img = tf.io.read_file(str(path))
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    img = tf.expand_dims(img, 0)
    vec = _get_model().predict(img, verbose=0)[0]
    return vec.astype(np.float32)


def register_dog(nose_image_path, dog_name, dog_id=None):
    emb = np.load(DB / "embeddings.npy")
    ids = np.load(DB / "dog_ids.npy")
    paths = np.load(DB / "paths.npy", allow_pickle=True)
    with open(DB / "metadata.json") as f:
        meta = json.load(f)

    if dog_id is None:
        dog_id = max(meta["dog_ids"]) + 1 if meta["dog_ids"] else 0
    dog_id = int(dog_id)

    vec = _embed(nose_image_path)

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
        "dog_id": dog_id,
        "dog_name": dog_name,
        "total_vectors": len(new_emb),
        "total_dogs": meta["total_dogs"],
    }