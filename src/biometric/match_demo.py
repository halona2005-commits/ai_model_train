from pathlib import Path
import numpy as np
import random

ROOT = Path(r"D:\internship_RIED\ai-model")
EMB = ROOT / "dataset" / "biometric" / "embeddings"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TOP_K = 5
N_QUERIES_TO_SHOW = 10

# ---------------- LOAD ----------------
print("Loading embeddings...")
train_emb    = np.load(EMB / "train_embeddings.npy")
train_labels = np.load(EMB / "train_labels.npy")
train_paths  = np.load(EMB / "train_paths.npy", allow_pickle=True)

enroll_emb    = np.load(EMB / "enrollment_embeddings.npy")
enroll_labels = np.load(EMB / "enrollment_labels.npy")
enroll_paths  = np.load(EMB / "enrollment_paths.npy", allow_pickle=True)

# ---------------- BUILD QUERY + GALLERY ----------------
# For each train dog with >=2 images, hold out 1 image as query.
# Everything else (rest of train + all enrollment) becomes gallery.

by_dog = {}
for emb, lbl, path in zip(train_emb, train_labels, train_paths):
    by_dog.setdefault(int(lbl), []).append((emb, lbl, path))

query_embs, query_labels, query_paths = [], [], []
gallery_embs, gallery_labels, gallery_paths = [], [], []

for dog_id, items in by_dog.items():
    random.shuffle(items)
    query_embs.append(items[0][0]); query_labels.append(items[0][1]); query_paths.append(items[0][2])
    for e, l, p in items[1:]:
        gallery_embs.append(e); gallery_labels.append(l); gallery_paths.append(p)

# add enrollment to gallery
for e, l, p in zip(enroll_emb, enroll_labels, enroll_paths):
    gallery_embs.append(e); gallery_labels.append(l); gallery_paths.append(p)

query_emb    = np.stack(query_embs)
query_labels = np.array(query_labels)
query_paths  = np.array(query_paths)
gallery_emb  = np.stack(gallery_embs)
gallery_labels = np.array(gallery_labels)
gallery_paths = np.array(gallery_paths)

print(f"Queries : {query_emb.shape}  ({len(set(query_labels))} unique dogs)")
print(f"Gallery : {gallery_emb.shape}  ({len(set(gallery_labels))} unique dogs)")
print(f"Dogs appearing in BOTH query and gallery: "
      f"{len(set(query_labels) & set(gallery_labels))}")

# ---------------- SIMILARITY ----------------
print("\nComputing similarity matrix...")
sim = query_emb @ gallery_emb.T   # [N_q, N_g]
top_idx = np.argsort(-sim, axis=1)

# ---------------- TOP-K ACCURACY ----------------
correct_top1 = 0
correct_top5 = 0
total = len(query_labels)
ranks = []

for i in range(total):
    true_id = query_labels[i]
    top5_ids = gallery_labels[top_idx[i, :TOP_K]]
    if top5_ids[0] == true_id:
        correct_top1 += 1
    if true_id in top5_ids:
        correct_top5 += 1
    # rank of the correct match
    pos = np.where(gallery_labels[top_idx[i]] == true_id)[0]
    if len(pos) > 0:
        ranks.append(pos[0] + 1)

ranks = np.array(ranks)

print()
print("=" * 60)
print("MATCHING ACCURACY (query = 1 image/dog, gallery = rest)")
print("=" * 60)
print(f"Top-1 accuracy: {correct_top1}/{total} = {100*correct_top1/total:.2f}%")
print(f"Top-5 accuracy: {correct_top5}/{total} = {100*correct_top5/total:.2f}%")
if len(ranks):
    print(f"Median rank of correct match: {int(np.median(ranks))}")

# ---------------- SAMPLE QUERIES ----------------
print()
print("=" * 60)
print(f"SAMPLE QUERIES ({N_QUERIES_TO_SHOW})")
print("=" * 60)

sample_idx = random.sample(range(total), min(N_QUERIES_TO_SHOW, total))
for i in sample_idx:
    true_id = query_labels[i]
    print(f"\nQuery: {Path(query_paths[i]).name}  (true dog_id={true_id})")
    print("-" * 55)
    for rank in range(TOP_K):
        j = top_idx[i, rank]
        matched_id = gallery_labels[j]
        mark = "✅" if matched_id == true_id else ""
        print(f"  #{rank+1}  dog_id={matched_id:<5}  sim={sim[i,j]:.4f}  {mark}")

# ---------------- SIMILARITY DISTRIBUTION ----------------
print()
print("=" * 60)
print("SIMILARITY DISTRIBUTION")
print("=" * 60)
same_sims = []
diff_sims = []
for i in range(total):
    true_id = query_labels[i]
    mask_same = gallery_labels == true_id
    same_sims.extend(sim[i, mask_same].tolist())
    diff_sims.extend(sim[i, ~mask_same].tolist())

same_sims = np.array(same_sims)
diff_sims = np.array(diff_sims)
print(f"Same-dog  pairs: n={len(same_sims):>7}  mean={same_sims.mean():.3f}  median={np.median(same_sims):.3f}")
print(f"Diff-dog  pairs: n={len(diff_sims):>7}  mean={diff_sims.mean():.3f}  median={np.median(diff_sims):.3f}")