from pathlib import Path
import numpy as np
import json

ROOT = Path(r"D:\internship_RIED\ai-model")
EMB = ROOT / "dataset" / "biometric" / "embeddings"
DB = ROOT / "dataset" / "biometric" / "database"
DB.mkdir(parents=True, exist_ok=True)

print("Loading existing embeddings...")
train_emb  = np.load(EMB / "train_embeddings.npy")
train_lbl  = np.load(EMB / "train_labels.npy")
train_pth  = np.load(EMB / "train_paths.npy", allow_pickle=True)

enroll_emb = np.load(EMB / "enrollment_embeddings.npy")
enroll_lbl = np.load(EMB / "enrollment_labels.npy")
enroll_pth = np.load(EMB / "enrollment_paths.npy", allow_pickle=True)

all_emb = np.concatenate([train_emb, enroll_emb], axis=0).astype(np.float32)
all_lbl = np.concatenate([train_lbl, enroll_lbl], axis=0).astype(np.int32)
all_pth = np.concatenate([train_pth, enroll_pth], axis=0)

np.save(DB / "embeddings.npy", all_emb)
np.save(DB / "dog_ids.npy", all_lbl)
np.save(DB / "paths.npy", all_pth)

unique_dogs = sorted(int(d) for d in np.unique(all_lbl))
metadata = {
    "total_vectors": int(len(all_emb)),
    "total_dogs": len(unique_dogs),
    "dog_ids": unique_dogs,
    "embedding_dim": int(all_emb.shape[1]),
}
with open(DB / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Database built at: {DB}")
print(f"  Vectors: {len(all_emb)}")
print(f"  Dogs:    {len(unique_dogs)}")
print(f"  Dim:     {all_emb.shape[1]}")