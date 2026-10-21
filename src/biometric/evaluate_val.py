"""
Evaluate v2 model on VALIDATION set only.

Purpose: choose a match/reject threshold that satisfies a target FAR.

Rules from plan:
    - Do NOT use test set here
    - Only cross-cluster pairs count as genuine matches
    - Report ROC-AUC, genuine/impostor distributions, FAR/FRR at candidate thresholds
"""

from pathlib import Path
import numpy as np

ROOT = Path(r"D:\internship_RIED\ai-model")
EMB = ROOT / "dataset_corrected" / "embeddings_v2"
OUT = ROOT / "dataset_corrected" / "eval"
OUT.mkdir(parents=True, exist_ok=True)

# Load val embeddings
emb = np.load(EMB / "val_embeddings.npy")
labels = np.load(EMB / "val_labels.npy")
clusters = np.load(EMB / "val_clusters.npy")

print(f"Val set: {emb.shape}")
print(f"Unique dogs: {len(set(labels.tolist()))}")
print()

# Similarity matrix
sim = emb @ emb.T  # cosine, since embeddings are L2-normalized

# Build genuine/impostor pairs
genuine = []
impostor = []
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        same_dog = labels[i] == labels[j]
        same_cluster = clusters[i] == clusters[j]
        s = float(sim[i, j])
        if same_dog and not same_cluster:
            genuine.append(s)
        elif not same_dog:
            impostor.append(s)
        # same_dog AND same_cluster -> skip (that's not a real match test)

genuine = np.array(genuine)
impostor = np.array(impostor)

print(f"Genuine pairs (same dog, different cluster): {len(genuine)}")
print(f"Impostor pairs (different dog):              {len(impostor)}")
print()

print("=" * 60)
print("SCORE DISTRIBUTIONS")
print("=" * 60)
print(f"Genuine  mean={genuine.mean():.3f}  median={np.median(genuine):.3f}  "
      f"p5={np.percentile(genuine,5):.3f}  p95={np.percentile(genuine,95):.3f}")
print(f"Impostor mean={impostor.mean():.3f}  median={np.median(impostor):.3f}  "
      f"p5={np.percentile(impostor,5):.3f}  p95={np.percentile(impostor,95):.3f}")

# ROC-AUC
try:
    from sklearn.metrics import roc_auc_score
    y_true = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
    y_score = np.concatenate([genuine, impostor])
    auc = roc_auc_score(y_true, y_score)
    print(f"\nROC-AUC: {auc:.4f}")
except ImportError:
    auc = None
    print("sklearn not available - skipping ROC-AUC")

# Threshold sweep
print()
print("=" * 60)
print("THRESHOLD SWEEP")
print("=" * 60)
print(f"{'thresh':>7} {'FAR':>8} {'FRR':>8} {'TAR':>8}")

targets = [0.01, 0.001, 0.0001]
results = []

for t in np.arange(0.30, 0.91, 0.05):
    far = float((impostor >= t).mean())
    frr = float((genuine < t).mean())
    tar = 1 - frr
    results.append((t, far, frr, tar))
    print(f"{t:>7.2f} {far:>8.4f} {frr:>8.4f} {tar:>8.4f}")

# Recommend a threshold at FAR=0.01 (1%)
best_t = None
for t, far, frr, tar in results:
    if far <= 0.01:
        best_t = t
        print()
        print(f"Recommended threshold (FAR<=1%): {t:.2f}")
        print(f"  FAR on val: {far*100:.3f}%")
        print(f"  FRR on val: {frr*100:.3f}%")
        print(f"  TAR on val: {tar*100:.3f}%")
        break

if best_t is None:
    print("\nNo threshold achieves FAR<=1% on val. Dataset too small or model too weak.")

# Save
np.save(OUT / "val_genuine_scores.npy", genuine)
np.save(OUT / "val_impostor_scores.npy", impostor)
(OUT / "val_threshold.txt").write_text(
    f"Recommended threshold: {best_t}\n"
    f"FAR on val: {far if best_t else 'N/A'}\n"
    f"FRR on val: {frr if best_t else 'N/A'}\n"
    f"ROC-AUC: {auc}\n"
)
print()
print("Outputs in:", OUT)