from pathlib import Path
import numpy as np
import json
from src.pipeline.prepare_nose import prepare_nose
from src.pipeline.recognize import _embed_crop

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "dataset" / "biometric" / "database"

# Default duplicate threshold (Backend should pass 0.65 or higher to prevent same-breed collisions)
DEFAULT_DUPLICATE_THRESHOLD = 0.65

def register_dog(nose_image_path, dog_name, dog_id=None, duplicate_threshold=DEFAULT_DUPLICATE_THRESHOLD):
    prep = prepare_nose(image_path=nose_image_path)
    if prep["status"] != "accepted":
        return {
            "status": "rejected",
            "reason_code": prep["reason_code"],
            "message": f"Cannot register: {prep['reason_code']}",
        }

    vec = _embed_crop(prep["crop_bgr"])

    emb = np.load(DB / "embeddings.npy")
    ids = np.load(DB / "dog_ids.npy")
    paths = np.load(DB / "paths.npy", allow_pickle=True)
    with open(DB / "metadata.json") as f:
        meta = json.load(f)

    # CHECK FOR DUPLICATES using dynamic threshold
    if len(emb) > 0:
        sims = emb @ vec
        max_sim = float(np.max(sims))
        
        if max_sim >= duplicate_threshold:
            best_match_idx = int(np.argmax(sims))
            existing_dog_id = int(ids[best_match_idx])
            
            # Also grab the Top 5 for debugging
            top_idx = np.argsort(-sims)[:5]
            top_matches = [{"dog_id": int(ids[i]), "similarity": float(sims[i])} for i in top_idx]
            
            return {
                "status": "already_registered",
                "dog_id": existing_dog_id,
                "similarity": max_sim,
                "message": f"This dog is already registered (Similarity: {max_sim:.2f})",
                "top5": top_matches
            }

    # ... (rest of the save logic remains exactly the same) ...
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

    return {"status": "registered", "dog_id": dog_id, "dog_name": dog_name}