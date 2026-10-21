"""
FINAL evaluation on the locked test set.
Run ONCE. Do not tune based on results.

Two protocols:
    A) Genuine match test — among test dogs only (multi-cluster)
         Enroll 1 cluster per test dog
         Query other clusters of same dog
         Expected: MATCHED

    B) Unknown rejection test
         Gallery = train + val embeddings
         Query = test images (all test dogs are unseen)
         Expected: UNKNOWN

User sets THRESHOLD below. Everything else locked.
"""

from pathlib import Path
import numpy as np

ROOT = Path(r"D:\internship_RIED\ai-model")
EMB = ROOT / "dataset_corrected" / "embeddings_v2"
OUT = ROOT / "dataset_corrected" / "eval"
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# USER: change this ONE value after picking on validation
# ============================================================
THRESHOLD = 0.70
# ============================================================


def load(name):
    return (
        np.load(EMB / f"{name}_embeddings.npy"),
        np.load(EMB / f"{name}_labels.npy"),
        np.load(EMB / f"{name}_clusters.npy"),
    )

train_emb, train_lbl, train_clu = load("train")
val_emb,   val_lbl,   val_clu   = load("val")
test_emb,  test_lbl,  test_clu  = load("test")

print(f"Threshold (locked): {THRESHOLD}")
print(f"Train : {train_emb.shape}  dogs={len(set(train_lbl.tolist()))}")
print(f"Val   : {val_emb.shape}  dogs={len(set(val_lbl.tolist()))}")
print(f"Test  : {test_emb.shape}  dogs={len(set(test_lbl.tolist()))}")
print()

# ============================================================
# PROTOCOL A — Genuine match test (test dogs, cross-cluster)
# ============================================================
print("=" * 60)
print("PROTOCOL A — Genuine match test")
print("=" * 60)
print("For each test dog with >=2 clusters:")
print("  Enroll 1 image from cluster A")
print("  Query with each image from cluster B (different cluster)")
print("  Expected: matched at threshold")
print()

# Build per-dog cluster map for test
test_by_dog = {}
for i in range(len(test_lbl)):
    dog = int(test_lbl[i])
    clu = int(test_clu[i])
    test_by_dog.setdefault(dog, {}).setdefault(clu, []).append(i)

# Collect enrollment (cluster 0) and queries (other clusters) per dog
enroll_list = []
query_list = []   # (emb_idx, dog_id)
for dog, clusters in test_by_dog.items():
    if len(clusters) < 2:
        continue
    cids = sorted(clusters.keys())
    enroll_cid = cids[0]
    enroll_list.append((dog, clusters[enroll_cid][0]))   # take 1 image
    for cid in cids[1:]:
        for idx in clusters[cid]:
            query_list.append((dog, idx))

print(f"Test dogs with >=2 clusters : {sum(1 for c in test_by_dog.values() if len(c) >= 2)}")
print(f"Enrollment images           : {len(enroll_list)}")
print(f"Query images                : {len(query_list)}")
print()

# Build gallery from enrollments
gallery_emb  = np.stack([test_emb[i] for _, i in enroll_list])
gallery_dogs = np.array([d for d, _ in enroll_list])

genuine_correct = 0
genuine_wrong   = 0
genuine_unknown = 0
wrong_matched_to = []

for true_dog, q_idx in query_list:
    q = test_emb[q_idx]
    sims = gallery_emb @ q
    top_i = int(np.argmax(sims))
    top_sim = float(sims[top_i])
    top_dog = int(gallery_dogs[top_i])

    if top_sim < THRESHOLD:
        genuine_unknown += 1
    elif top_dog == true_dog:
        genuine_correct += 1
    else:
        genuine_wrong += 1
        wrong_matched_to.append((true_dog, top_dog, top_sim))

total_q = len(query_list)
print(f"Genuine queries             : {total_q}")
print(f"  Correct match             : {genuine_correct}  ({100*genuine_correct/total_q:.2f}%)")
print(f"  Wrong match               : {genuine_wrong}  ({100*genuine_wrong/total_q:.2f}%)")
print(f"  Rejected (below thresh)   : {genuine_unknown}  ({100*genuine_unknown/total_q:.2f}%)")
print()

# ============================================================
# PROTOCOL B — Unknown rejection test
# ============================================================
print("=" * 60)
print("PROTOCOL B — Unknown-dog rejection")
print("=" * 60)
print("Gallery = train + val embeddings")
print("Query   = all test images (test dogs are unseen)")
print("Expected: UNKNOWN at threshold")
print()

# Build gallery
gallery_emb2 = np.concatenate([train_emb, val_emb], axis=0)
gallery_lbl2 = np.concatenate([train_lbl, val_lbl], axis=0)

false_accept = 0
correct_reject = 0
false_accept_examples = []

for i in range(len(test_emb)):
    q = test_emb[i]
    sims = gallery_emb2 @ q
    top_sim = float(sims.max())
    top_i = int(np.argmax(sims))
    top_dog = int(gallery_lbl2[top_i])

    if top_sim >= THRESHOLD:
        false_accept += 1
        if len(false_accept_examples) < 5:
            false_accept_examples.append((int(test_lbl[i]), top_dog, top_sim))
    else:
        correct_reject += 1

total_u = len(test_emb)
print(f"Unknown queries             : {total_u}")
print(f"  Correctly rejected        : {correct_reject}  ({100*correct_reject/total_u:.2f}%)")
print(f"  False accepted            : {false_accept}  ({100*false_accept/total_u:.2f}%)")
print()

if false_accept_examples:
    print("Example false accepts (test_dog -> accepted_as, similarity):")
    for td, ad, s in false_accept_examples:
        print(f"  true_dog={td:<6} accepted_as={ad:<6} sim={s:.4f}")
    print()

# ============================================================
# ROC-AUC on test (genuine vs impostor pairs)
# ============================================================
print("=" * 60)
print("ROC-AUC on test")
print("=" * 60)

sim = test_emb @ test_emb.T
genuine = []
impostor = []
for i in range(len(test_lbl)):
    for j in range(i + 1, len(test_lbl)):
        same_dog = test_lbl[i] == test_lbl[j]
        same_clu = test_clu[i] == test_clu[j]
        s = float(sim[i, j])
        if same_dog and not same_clu:
            genuine.append(s)
        elif not same_dog:
            impostor.append(s)

genuine = np.array(genuine)
impostor = np.array(impostor)

try:
    from sklearn.metrics import roc_auc_score
    y_true = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
    y_score = np.concatenate([genuine, impostor])
    auc = roc_auc_score(y_true, y_score)
    print(f"ROC-AUC (test): {auc:.4f}")
    print(f"  genuine pairs : {len(genuine)}  mean={genuine.mean():.3f}")
    print(f"  impostor pairs: {len(impostor)}  mean={impostor.mean():.3f}")
except ImportError:
    print("sklearn not installed")

# ============================================================
# SAVE
# ============================================================
report = []
report.append("=" * 60)
report.append("FINAL LOCKED TEST EVALUATION")
report.append("=" * 60)
report.append(f"Threshold         : {THRESHOLD}")
report.append(f"Test images       : {len(test_emb)}")
report.append(f"Test dogs         : {len(set(test_lbl.tolist()))}")
report.append("")
report.append("PROTOCOL A — Genuine match")
report.append(f"  queries         : {total_q}")
report.append(f"  correct match   : {genuine_correct}  ({100*genuine_correct/total_q:.2f}%)")
report.append(f"  wrong match     : {genuine_wrong}")
report.append(f"  rejected        : {genuine_unknown}")
report.append("")
report.append("PROTOCOL B — Unknown rejection")
report.append(f"  queries         : {total_u}")
report.append(f"  correctly reject: {correct_reject}  ({100*correct_reject/total_u:.2f}%)")
report.append(f"  false accept    : {false_accept}  ({100*false_accept/total_u:.2f}%)")
report.append("")
try:
    report.append(f"ROC-AUC (test)    : {auc:.4f}")
except NameError:
    pass

(OUT / "final_test_report.txt").write_text("\n".join(report))
print()
print("Report saved to:", OUT / "final_test_report.txt")