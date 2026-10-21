"""
Build identity-disjoint manifest with cluster IDs and locked splits.

Reads:
    dataset_corrected/audit/hashes.csv
    dataset_corrected/train/train_data_corrected.csv

Writes:
    dataset_corrected/manifest.csv
    dataset_corrected/manifest_summary.txt

Manifest columns:
    image, dog_id, cluster_id, md5, phash, split, keep

Rules:
    - Exact duplicates (same md5): keep=1 for first occurrence, keep=0 for rest
    - cluster_id: per-dog integer, images in same pHash cluster share a value
    - split: 'train' | 'val' | 'test' assigned at DOG level
    - All images of a dog go to the same split
"""

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import imagehash


# ============================================================
# CONFIG
# ============================================================
ROOT = Path(r"D:\internship_RIED\ai-model")
DATASET = ROOT / "dataset_corrected" / "train"
AUDIT = ROOT / "dataset_corrected" / "audit"
OUT = ROOT / "dataset_corrected"

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

SAME_SESSION_THRESHOLD = 12


# ============================================================
# LOAD
# ============================================================
print("Loading audit data...")
hashes = pd.read_csv(AUDIT / "hashes.csv")
print(f"  Hashes rows : {len(hashes)}")
print(f"  Unique dogs : {hashes['dog_id'].nunique()}")

# ============================================================
# ASSIGN CLUSTER IDs PER DOG
# ============================================================
print("\nAssigning cluster IDs (per dog)...")

hashes["cluster_id"] = -1

for dog_id, group in hashes.groupby("dog_id"):
    idxs = group.index.tolist()
    phashes = [imagehash.hex_to_hash(p) if isinstance(p, str) else None
               for p in group["phash"]]

    n = len(idxs)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        if phashes[i] is None:
            continue
        for j in range(i + 1, n):
            if phashes[j] is None:
                continue
            if (phashes[i] - phashes[j]) <= SAME_SESSION_THRESHOLD:
                union(i, j)

    # Convert union-find roots to compact cluster IDs 0, 1, 2, ...
    roots = [find(i) for i in range(n)]
    unique_roots = sorted(set(roots))
    root_to_cluster = {r: c for c, r in enumerate(unique_roots)}

    for i, idx in enumerate(idxs):
        hashes.at[idx, "cluster_id"] = root_to_cluster[roots[i]]

print(f"  Cluster IDs assigned.")

# ============================================================
# MARK DUPLICATES (keep = 1 for first, 0 for rest)
# ============================================================
print("\nMarking exact duplicates...")
hashes["keep"] = 1

# Group by md5, keep first row, mark rest as keep=0
seen_md5 = set()
for i, row in hashes.iterrows():
    if row["md5"] in seen_md5:
        hashes.at[i, "keep"] = 0
    else:
        seen_md5.add(row["md5"])

dups = (hashes["keep"] == 0).sum()
print(f"  Duplicate files marked keep=0: {dups}")

# ============================================================
# SPLIT DOGS 70/15/15
# ============================================================
print("\nSplitting dogs (identity-disjoint)...")

# Only multi-cluster dogs are useful for anchor-positive training.
# We still include single-cluster dogs, but they only contribute to test.
dog_clusters = hashes.groupby("dog_id")["cluster_id"].nunique()
multi_dogs = sorted(dog_clusters[dog_clusters >= 2].index.tolist())
single_dogs = sorted(dog_clusters[dog_clusters == 1].index.tolist())

print(f"  Dogs with >=2 clusters : {len(multi_dogs)}")
print(f"  Dogs with 1 cluster    : {len(single_dogs)}")

rng = np.random.default_rng(SEED)
rng.shuffle(multi_dogs)

n = len(multi_dogs)
n_train = int(TRAIN_FRAC * n)
n_val = int(VAL_FRAC * n)

train_dogs = set(multi_dogs[:n_train])
val_dogs = set(multi_dogs[n_train:n_train + n_val])
test_dogs = set(multi_dogs[n_train + n_val:])

# Single-cluster dogs → test only (they cannot train, they can only be enrolled)
test_dogs |= set(single_dogs)

def split_for(dog_id):
    if dog_id in train_dogs:
        return "train"
    if dog_id in val_dogs:
        return "val"
    return "test"

hashes["split"] = hashes["dog_id"].apply(split_for)

# ============================================================
# WRITE MANIFEST
# ============================================================
manifest_path = OUT / "manifest.csv"
hashes[["image", "dog_id", "cluster_id", "md5", "phash", "split", "keep"]].to_csv(
    manifest_path, index=False
)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("MANIFEST SUMMARY")
print("=" * 60)

summary_lines = []
def log(s):
    print(s)
    summary_lines.append(s)

log(f"Manifest rows     : {len(hashes)}")
log(f"Kept rows         : {(hashes['keep'] == 1).sum()}")
log(f"Dropped dupes     : {(hashes['keep'] == 0).sum()}")
log(f"Unique dogs       : {hashes['dog_id'].nunique()}")
log("")

for split in ["train", "val", "test"]:
    sub = hashes[hashes["split"] == split]
    kept = sub[sub["keep"] == 1]
    log(f"[{split}]")
    log(f"  dogs      : {sub['dog_id'].nunique()}")
    log(f"  images    : {len(sub)}  (kept: {len(kept)})")
    log(f"  clusters  : {sub['cluster_id'].nunique()}")
    log("")

# SHA-256 of manifest itself
manifest_bytes = manifest_path.read_bytes()
manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
log(f"Manifest SHA-256  : {manifest_sha}")

summary_path = OUT / "manifest_summary.txt"
summary_path.write_text("\n".join(summary_lines))

print()
print("Outputs:")
print(f"  {manifest_path}")
print(f"  {summary_path}")
print()
print(f"LOCKED. Manifest SHA-256: {manifest_sha}")
print("Do not modify this file. All downstream scripts read from it.")