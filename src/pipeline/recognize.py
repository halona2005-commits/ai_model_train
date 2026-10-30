from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

from src.pipeline.prepare_nose import prepare_nose

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "dataset" / "biometric" / "database"
MODEL_BIOMETRIC = ROOT / "models" / "biometric" / "inference_embedding_model.keras"

IMG_SIZE = (224, 224)
MATCH_THRESHOLD = 0.80

_biometric = None


def _get_model():
    global _biometric
    if _biometric is None:
        _biometric = tf.keras.models.load_model(MODEL_BIOMETRIC, compile=False)
    return _biometric


def _embed_crop(crop_bgr):
    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32)
    img = preprocess_input(img)
    vec = _get_model().predict(np.expand_dims(img, 0), verbose=0)[0]
    return vec.astype(np.float32)


def recognize(image_path, top_k=5):
    prep = prepare_nose(image_path=image_path)
    if prep["status"] != "accepted":
        return {
            "status": "not_a_nose",
            "reason_code": prep["reason_code"],
            "validator_score": prep["validator_score"],
            "message": f"Rejected: {prep['reason_code']}",
        }

    vec = _embed_crop(prep["crop_bgr"])

    db_emb = np.load(DB / "embeddings.npy")
    db_ids = np.load(DB / "dog_ids.npy")

    sims = db_emb @ vec
    top_idx = np.argsort(-sims)[:top_k]
    top_matches = [
        {"dog_id": int(db_ids[i]), "similarity": float(sims[i])}
        for i in top_idx
    ]
    best = top_matches[0]

    if best["similarity"] >= MATCH_THRESHOLD:
        return {
            "status": "matched",
            "dog_id": best["dog_id"],
            "confidence": best["similarity"],
            "validator_score": prep["validator_score"],
            "used_yolo": prep["used_yolo"],
            "top5": top_matches,
        }
    else:
        return {
            "status": "unknown",
            "message": "No matching dog. Register as new?",
            "best_guess": best,
            "validator_score": prep["validator_score"],
            "used_yolo": prep["used_yolo"],
            "top5": top_matches,
        }