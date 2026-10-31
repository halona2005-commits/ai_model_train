import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import random
from pathlib import Path
from sklearn.metrics import roc_curve, auc, accuracy_score
from src.pipeline.prepare_nose import prepare_nose

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(r"D:\internship_RIED\ai-model")
EMB_DIR = BASE_DIR / "embeddings"  # MUST CONTAIN THE TEAM'S .NPY FILES
SPLITS_DIR = BASE_DIR / "pet_biometric_full" / "splits"
IMG_DIR = BASE_DIR / "pet_biometric_full" / "train" / "images"
RESULTS_DIR = BASE_DIR / "results"

if not EMB_DIR.exists():
    raise FileNotFoundError(f"Missing embeddings folder! Please put the team's '.npy' files in {EMB_DIR}")

print("Creating results folder...")
if RESULTS_DIR.exists():
    shutil.rmtree(RESULTS_DIR)
(RESULTS_DIR / "embeddings").mkdir(parents=True)

# ============================================================
# 1. COPY FILES & CREATE ENROLLMENT CSV
# ============================================================
print("1. Copying CSVs and embeddings...")
for file in EMB_DIR.glob("*.npy"):
    shutil.copy(file, RESULTS_DIR / "embeddings" / file.name)

train_df = pd.read_csv(SPLITS_DIR / "train.csv")
val_df = pd.read_csv(SPLITS_DIR / "val.csv")
test_df = pd.read_csv(SPLITS_DIR / "test.csv")

train_df.to_csv(RESULTS_DIR / "train.csv", index=False)
val_df.to_csv(RESULTS_DIR / "val.csv", index=False)
test_df.to_csv(RESULTS_DIR / "test.csv", index=False)

# Create enrollment.csv (1 image per test dog)
enrollment_df = test_df.drop_duplicates(subset=["dog_id"]).copy()
enrollment_df.to_csv(RESULTS_DIR / "enrollment.csv", index=False)

# ============================================================
# 2. CALCULATE METRICS
# ============================================================
print("2. Calculating Metrics...")
val_emb = np.load(EMB_DIR / "val_embeddings.npy")
val_lbl = np.load(EMB_DIR / "val_labels.npy")
test_emb = np.load(EMB_DIR / "test_embeddings.npy")
test_lbl = np.load(EMB_DIR / "test_labels.npy")

def get_pairs(emb, lbl):
    unique_dogs = np.unique(lbl)
    genuine, impostor = [], []
    for dog in unique_dogs:
        idxs = np.where(lbl == dog)[0]
        if len(idxs) > 1:
            for i in range(1, len(idxs)):
                genuine.append(np.dot(emb[idxs[0]], emb[idxs[i]]))
            diff_dogs = unique_dogs[unique_dogs != dog]
            if len(diff_dogs) > 0:
                diff_idx = np.where(lbl == diff_dogs[0])[0][0]
                impostor.append(np.dot(emb[idxs[0]], emb[diff_idx]))
    return np.array(genuine), np.array(impostor)

# Validation (Find Threshold at 1% FAR)
val_gen, val_imp = get_pairs(val_emb, val_lbl)
y_true = np.concatenate([np.ones_like(val_gen), np.zeros_like(val_imp)])
y_scores = np.concatenate([val_gen, val_imp])
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

valid_idx = np.where(fpr <= 0.01)[0]
locked_thresh = thresholds[valid_idx[-1]] if len(valid_idx) > 0 else 0.7

# Test (Calculate Rank-1, FAR, FRR)
unique_test = np.unique(test_lbl)
gallery, gallery_lbl, query, query_lbl = [], [], [], []
for dog in unique_test:
    idxs = np.where(test_lbl == dog)[0]
    if len(idxs) > 0:
        gallery.append(test_emb[idxs[0]])
        gallery_lbl.append(dog)
        for i in range(1, len(idxs)):
            query.append(test_emb[idxs[i]])
            query_lbl.append(dog)

gallery = np.array(gallery)
gallery_lbl = np.array(gallery_lbl)
query = np.array(query)
query_lbl = np.array(query_lbl)

sim_matrix = query @ gallery.T
top1_preds = gallery_lbl[np.argmax(sim_matrix, axis=1)]
rank1 = accuracy_score(query_lbl, top1_preds)

test_gen, test_imp = get_pairs(test_emb, test_lbl)
frr = np.mean(test_gen < locked_thresh)
far = np.mean(test_imp >= locked_thresh)

# ============================================================
# 3. CREATE SUMMARY.JSON
# ============================================================
print("3. Creating summary.json...")
summary = {
    "model_architecture": "ResNet50 + ArcFace",
    "validation_set": {
        "roc_auc": round(float(roc_auc), 4),
        "locked_threshold_far_1pct": round(float(locked_thresh), 4)
    },
    "locked_test_set": {
        "rank_1_accuracy_pct": round(float(rank1 * 100), 2),
        "false_accept_rate_pct": round(float(far * 100), 2),
        "false_reject_rate_pct": round(float(frr * 100), 2)
    },
    "integration_files": {
        "video_processing": "video_frame_selector.py (included in this folder)"
    }
}
with open(RESULTS_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=4)

# ============================================================
# 4. GENERATE SMOKE TEST CROPS IMAGE
# ============================================================
print("4. Generating smoke_test_crops.png...")
random.seed(42)
sample_df = test_df.sample(min(6, len(test_df)))

plt.figure(figsize=(15, 5))
for i, (_, row) in enumerate(sample_df.iterrows()):
    img_path = IMG_DIR / row["image"]
    prep = prepare_nose(image_path=img_path)
    
    plt.subplot(1, 6, i + 1)
    if prep["status"] == "accepted":
        crop = cv2.cvtColor(prep["crop_bgr"], cv2.COLOR_BGR2RGB)
        plt.imshow(crop)
        plt.title(f"Dog {row['dog_id']}\nAccepted", color='green')
    else:
        plt.text(0.5, 0.5, "Rejected", ha='center', va='center')
        plt.title(f"Dog {row['dog_id']}\nRejected", color='red')
    plt.axis('off')
plt.tight_layout()
plt.savefig(RESULTS_DIR / "smoke_test_crops.png")
plt.close()

# ============================================================
# 5. GENERATE VIDEO FRAME SELECTOR SCRIPT
# ============================================================
print("5. Generating video_frame_selector.py for WhatsApp Team...")
video_script = '''import cv2
import tempfile
from pathlib import Path
from src.pipeline.recognize import recognize
from src.pipeline.prepare_nose import prepare_nose

def recognize_from_video(video_path):
    """
    Extracts frames from a video, finds the one with the highest nose confidence,
    and runs the biometric recognition on it.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"status": "error", "message": "Could not open video"}

    # Sample 1 frame every 0.5 seconds
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(fps / 2) if fps > 0 else 15
    
    best_score = -1.0
    best_frame = None
    frame_count = 0

    with tempfile.TemporaryDirectory() as temp_dir:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_skip == 0:
                temp_path = Path(temp_dir) / f"frame_{frame_count}.jpg"
                cv2.imwrite(str(temp_path), frame)
                
                # Check if it's a valid nose
                prep = prepare_nose(image_path=temp_path)
                if prep["status"] == "accepted":
                    if prep["validator_score"] > best_score:
                        best_score = prep["validator_score"]
                        best_frame_path = temp_path
                        
            frame_count += 1
        cap.release()

        if best_frame and best_frame_path.exists():
            # Run full recognition on the best frame found
            result = recognize(best_frame_path)
            return result
        else:
            return {"status": "not_a_nose", "message": "No valid dog nose found in video"}

# Example usage:
# result = recognize_from_video("whatsapp_video.mp4")
'''
with open(RESULTS_DIR / "video_frame_selector.py", "w") as f:
    f.write(video_script)

print("\n" + "="*50)
print(f"✅ ALL DONE! Results folder created at:\n{RESULTS_DIR}")
print("="*50)
print(f"-> Locked Threshold: {locked_thresh:.4f}")
print(f"-> Test Rank-1 Accuracy: {rank1*100:.2f}%")
print(f"-> False Accept Rate: {far*100:.2f}%")
print("\nZip the 'results' folder and send it to the team.")