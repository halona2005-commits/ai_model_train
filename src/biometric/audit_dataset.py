"""
Audit the nose-print dataset for duplicates and session leakage.

Outputs:
    dataset_corrected/audit/hashes.csv         — md5 + phash per image
    dataset_corrected/audit/report.txt         — human-readable summary
    dataset_corrected/audit/dog_summary.csv    — per-dog stats
"""

from pathlib import Path
import hashlib
from collections import defaultdict

import pandas as pd
import numpy as np
from PIL import Image
import imagehash


# ============================================================
# CONFIG  — pointing at dataset_corrected
# ============================================================
DATASET = Path(r"D:\internship_RIED\ai-model\dataset_corrected\train")
IMAGES  = DATASET / "images"
CSV     = DATASET / "train_data_corrected.csv"
OUT     = Path(r"D:\internship_RIED\ai-model\dataset_corrected\audit")
OUT.mkdir(parents=True, exist_ok=True)

# pHash Hamming distance thresholds
NEAR_DUP_THRESHOLD     = 10
SAME_SESSION_THRESHOLD = 12

# ============================================================
# LOAD
# ============================================================
df = pd.read_csv(CSV)
df.columns = ["dog_id", "image"]
print(f"Dataset rows  : {len(df)}")
print(f"Unique dogs   : {df['dog_id'].nunique()}")
print(f"Images folder : {IMAGES}")
print()

# ============================================================
# COMPUTE HASHES
# ============================================================
records = []
missing = 0
for _, row in df.iterrows():
    p = IMAGES / row["image"]
    if not p.exists():
        missing += 1
        continue

    md5 = hashlib.md5(p.read_bytes()).hexdigest()
    try:
        with Image.open(p) as im:
            ph = imagehash.phash(im)
    except Exception:
        ph = None

    records.append({
        "image": row["image"],
        "dog_id": int(row["dog_id"]),
        "md5": md5,
        "phash": str(ph) if ph else None,
    })

audit_df = pd.DataFrame(records)
audit_df.to_csv(OUT / "hashes.csv", index=False)

print(f"Hashed images : {len(audit_df)}  (missing on disk: {missing})")
print()

# ============================================================
# EXACT DUPLICATES
# ============================================================
md5_groups = defaultdict(list)
for _, r in audit_df.iterrows():
    md5_groups[r["md5"]].append(r["image"])

exact_dups = {h: imgs for h, imgs in md5_groups.items() if len(imgs) > 1}

print("=" * 70)
print("EXACT DUPLICATES (identical bytes)")
print("=" * 70)
print(f"Duplicate groups: {len(exact_dups)}")
total_dupe_files = sum(len(g) - 1 for g in exact_dups.values())
print(f"Redundant files : {total_dupe_files}")
if exact_dups:
    print("First 5 groups:")
    for i, (h, imgs) in enumerate(list(exact_dups.items())[:5]):
        print(f"  {h[:8]}  {imgs}")

# ============================================================
# PER-DOG NEAR-DUPLICATE ANALYSIS
# ============================================================
print()
print("=" * 70)
print("PER-DOG ANALYSIS")
print("=" * 70)

dog_summary = []

for dog_id, group in audit_df.groupby("dog_id"):
    imgs = group["image"].tolist()
    phashes = [imagehash.hex_to_hash(p) for p in group["phash"] if p is not None]

    if len(phashes) < 2:
        dog_summary.append({
            "dog_id": dog_id,
            "n_images": len(imgs),
            "n_unique_phash_clusters": 0,
            "min_intra_distance": None,
            "max_intra_distance": None,
            "flagged": "single_or_unhashable",
        })
        continue

    n = len(phashes)
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            d = phashes[i] - phashes[j]
            distances.append(d)

    distances = np.array(distances)
    min_d = int(distances.min())
    max_d = int(distances.max())

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
        for j in range(i + 1, n):
            if (phashes[i] - phashes[j]) <= SAME_SESSION_THRESHOLD:
                union(i, j)

    clusters = len(set(find(i) for i in range(n)))

    flag = ""
    if clusters == 1 and n > 1:
        flag = "ALL_ONE_SESSION"
    elif min_d <= 5:
        flag = "HAS_NEAR_DUP"
    elif clusters < n:
        flag = "SOME_CLUSTERING"

    dog_summary.append({
        "dog_id": dog_id,
        "n_images": n,
        "n_unique_phash_clusters": clusters,
        "min_intra_distance": min_d,
        "max_intra_distance": max_d,
        "flagged": flag,
    })

summary_df = pd.DataFrame(dog_summary)
summary_df.to_csv(OUT / "dog_summary.csv", index=False)

total_dogs = len(summary_df)
dogs_all_one = (summary_df["flagged"] == "ALL_ONE_SESSION").sum()
dogs_has_near = (summary_df["flagged"] == "HAS_NEAR_DUP").sum()
dogs_some_cluster = (summary_df["flagged"] == "SOME_CLUSTERING").sum()
dogs_clean = total_dogs - dogs_all_one - dogs_has_near - dogs_some_cluster
dogs_single = (summary_df["flagged"] == "single_or_unhashable").sum()

print(f"Total dogs                : {total_dogs}")
print(f"  Single image / unhashable: {dogs_single}")
print(f"  All images one session  : {dogs_all_one}  <- WARNING")
print(f"  Has near-duplicates     : {dogs_has_near}")
print(f"  Some clustering         : {dogs_some_cluster}")
print(f"  Clean (all independent) : {dogs_clean}")
print()

print("Distribution of unique session clusters per dog:")
cluster_counts = summary_df["n_unique_phash_clusters"].value_counts().sort_index()
for k, v in cluster_counts.items():
    print(f"  {k} cluster(s): {v} dogs")

# ============================================================
# WRITE REPORT
# ============================================================
report = []
report.append("=" * 70)
report.append("DATASET AUDIT REPORT")
report.append("=" * 70)
report.append(f"Source CSV      : {CSV}")
report.append(f"Image folder    : {IMAGES}")
report.append(f"Total rows      : {len(df)}")
report.append(f"Total hashed    : {len(audit_df)}")
report.append(f"Missing on disk : {missing}")
report.append(f"Unique dogs     : {audit_df['dog_id'].nunique()}")
report.append("")
report.append("EXACT DUPLICATES (identical bytes)")
report.append(f"  Duplicate groups: {len(exact_dups)}")
report.append(f"  Redundant files : {total_dupe_files}")
report.append("")
report.append("PER-DOG SUMMARY")
report.append(f"  Total dogs                : {total_dogs}")
report.append(f"  Single image / unhashable : {dogs_single}")
report.append(f"  All images one session    : {dogs_all_one}")
report.append(f"  Has near-duplicates       : {dogs_has_near}")
report.append(f"  Some clustering           : {dogs_some_cluster}")
report.append(f"  Clean (all independent)   : {dogs_clean}")
report.append("")
report.append("INTERPRETATION")
report.append("  ALL_ONE_SESSION -> all images from this dog come from one burst/capture.")
report.append("  HAS_NEAR_DUP    -> some images are near-duplicates.")
report.append("  SOME_CLUSTERING -> partial clustering. Review manually.")
report.append("  Clean           -> all images are visually distinct captures.")
report.append("")

(OUT / "report.txt").write_text("\n".join(report))

print()
print("=" * 70)
print("OUTPUTS")
print("=" * 70)
print(f"  {OUT / 'hashes.csv'}")
print(f"  {OUT / 'dog_summary.csv'}")
print(f"  {OUT / 'report.txt'}")