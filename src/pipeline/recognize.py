from pathlib import Path
import numpy as np
import cv2
import os
import tempfile
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import preprocess_input

from src.pipeline.prepare_nose import prepare_nose

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "dataset" / "biometric" / "database"
MODEL_BIOMETRIC = ROOT / "models" / "biometric" / "inference_embedding_model.keras"

IMG_SIZE = (224, 224)
# Default threshold (fallback if not passed by WhatsApp backend)
DEFAULT_THRESHOLD = 0.55  

_biometric = None

class L2Norm(layers.Layer):
    def call(self, x):
        return tf.math.l2_normalize(x, axis=1)

def _get_model():
    global _biometric
    if _biometric is None:
        _biometric = tf.keras.models.load_model(
            MODEL_BIOMETRIC, compile=False, custom_objects={'L2Norm': L2Norm}
        )
    return _biometric

def _embed_crop(crop_bgr):
    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32)
    img = preprocess_input(img)
    vec = _get_model().predict(np.expand_dims(img, 0), verbose=0)[0]
    return vec.astype(np.float32)

def _match_embedding(vec, threshold, top_k=5):
    db_emb = np.load(DB / "embeddings.npy")
    db_ids = np.load(DB / "dog_ids.npy")

    sims = db_emb @ vec
    top_idx = np.argsort(-sims)[:top_k]
    top_matches = [
        {"dog_id": int(db_ids[i]), "similarity": float(sims[i])}
        for i in top_idx
    ]
    best = top_matches[0]

    if best["similarity"] >= threshold:
        return {
            "status": "matched",
            "dog_id": best["dog_id"],
            "confidence": best["similarity"],
            "top5": top_matches,
        }
    else:
        return {
            "status": "unknown",
            "message": "No matching dog. Register as new?",
            "best_guess": best,
            "top5": top_matches,
        }

def _recognize_video(video_path, threshold, top_k=5):
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"status": "error", "message": "Could not open video"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(fps / 2) if fps > 0 else 15 
    
    valid_frames_data = [] # List of tuples: (composite_score, embedding)
    frame_count = 0

    with tempfile.TemporaryDirectory() as temp_dir:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_skip == 0:
                temp_path = os.path.join(temp_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(temp_path, frame)
                
                prep = prepare_nose(image_path=temp_path)
                if prep["status"] == "accepted":
                    # Calculate sharpness
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                    composite = prep["validator_score"] * sharpness
                    
                    # Embed the crop
                    vec = _embed_crop(prep["crop_bgr"])
                    valid_frames_data.append((composite, vec))
                        
            frame_count += 1
        cap.release()

        if not valid_frames_data:
            return {"status": "not_a_nose", "message": "No valid sharp dog nose found in video"}

        # MULTI-FRAME AVERAGING
        # Sort by composite score (best first) and take Top 5
        valid_frames_data.sort(key=lambda x: x[0], reverse=True)
        top_5_vecs = [data[1] for data in valid_frames_data[:5]]
        
        # Average the embeddings
        avg_vec = np.mean(top_5_vecs, axis=0)
        # Re-normalize the averaged vector
        avg_vec = avg_vec / np.linalg.norm(avg_vec)
        
        print(f"Averaged {len(top_5_vecs)} frames for video matching.")
        
        result = _match_embedding(avg_vec, threshold, top_k)
        result["frames_averaged"] = len(top_5_vecs)
        return result

def recognize(image_path, threshold=DEFAULT_THRESHOLD, top_k=5):
    # 1. Route videos to the video handler
    ext = str(image_path).lower().split('.')[-1]
    if ext in ['mp4', 'avi', 'mov', 'mkv', 'webm']:
        return _recognize_video(image_path, threshold, top_k)

    # 2. Process standard images
    prep = prepare_nose(image_path=image_path)
    if prep["status"] != "accepted":
        return {
            "status": "not_a_nose",
            "reason_code": prep["reason_code"],
            "validator_score": prep["validator_score"],
            "message": f"Rejected: {prep['reason_code']}",
        }

    vec = _embed_crop(prep["crop_bgr"])
    result = _match_embedding(vec, threshold, top_k)
    result["validator_score"] = prep["validator_score"]
    return result