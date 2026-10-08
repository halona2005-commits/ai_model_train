from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import json
import random

ROOT = Path(r"D:\internship_RIED\ai-model")
MODEL_PATH = ROOT / "models" / "nose_detector" / "yolo11n_nose_tight" / "weights" / "best.pt"

DOGFLW_TEST_IMG = ROOT / "DogFLW" / "test" / "images"
DOGFLW_TEST_LBL = ROOT / "DogFLW" / "test" / "labels"
OUT_DIR = ROOT / "dataset" / "localization" / "test_images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 30
CONF_THRESHOLD = 0.25
SEED = 42

print("Loading model:", MODEL_PATH)
model = YOLO(str(MODEL_PATH))

img_files = sorted(DOGFLW_TEST_IMG.glob("*"))
random.seed(SEED)
sample = random.sample(img_files, min(N_SAMPLES, len(img_files)))

hits = 0
misses = 0
ious = []
centroid_dists = []

def gt_nose_box_from_landmarks(lbl_path, W, H):
    """DogFLW landmarks 0..13 = nose region."""
    if not lbl_path.exists():
        return None
    with open(lbl_path) as f:
        data = json.load(f)
    lms = np.array(data["landmarks"], dtype=np.float32)
    if lms.shape != (46, 2):
        return None
    nose = lms[0:14]
    mask = np.isfinite(nose).all(axis=1)
    nose = nose[mask]
    if nose.shape[0] < 2:
        return None
    x_min, y_min = nose.min(axis=0)
    x_max, y_max = nose.max(axis=0)
    bw, bh = x_max - x_min, y_max - y_min
    x1 = max(0, x_min - 0.15 * bw)
    y1 = max(0, y_min - 0.15 * bh)
    x2 = min(W, x_max + 0.15 * bw)
    y2 = min(H, y_max + 0.15 * bh)
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def iou(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    u = areaA + areaB - inter
    return inter / u if u > 0 else 0.0

print()
print(f"{'file':<40} {'conf':>6} {'iou':>6} {'dist':>7}")
print("-" * 65)

for p in sample:
    img = cv2.imread(str(p))
    if img is None:
        continue
    H, W = img.shape[:2]

    results = model(str(p), conf=CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes

    lbl = DOGFLW_TEST_LBL / (p.stem + ".json")
    gt = gt_nose_box_from_landmarks(lbl, W, H)

    if boxes is None or len(boxes) == 0:
        print(f"{p.name[:38]:<40} {'--':>6} {'--':>6} {'--':>7}  (no detection)")
        misses += 1
        continue

    confs = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    i = int(np.argmax(confs))
    pred = xyxy[i]
    conf = float(confs[i])
    hits += 1

    if gt is not None:
        iou_val = iou(pred, gt)
        ious.append(iou_val)
        pc_x = (pred[0] + pred[2]) / 2
        pc_y = (pred[1] + pred[3]) / 2
        gc_x = (gt[0] + gt[2]) / 2
        gc_y = (gt[1] + gt[3]) / 2
        d = float(np.hypot(pc_x - gc_x, pc_y - gc_y))
        centroid_dists.append(d)
        print(f"{p.name[:38]:<40} {conf:>6.2f} {iou_val:>6.2f} {d:>7.1f}")
    else:
        print(f"{p.name[:38]:<40} {conf:>6.2f} {'--':>6} {'--':>7}  (no GT)")

    vis = img.copy()
    x1, y1, x2, y2 = pred
    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(vis, f"pred {conf:.2f}", (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if gt is not None:
        cv2.rectangle(vis, (int(gt[0]), int(gt[1])), (int(gt[2]), int(gt[3])),
                      (255, 0, 0), 2)
        cv2.putText(vis, "GT nose", (int(gt[0]), int(gt[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.imwrite(str(OUT_DIR / f"test_{p.stem}.jpg"), vis)

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Images tested : {len(sample)}")
print(f"Hits          : {hits}")
print(f"Misses        : {misses}")
if ious:
    print(f"Mean IoU      : {np.mean(ious):.3f}")
    print(f"Median IoU    : {np.median(ious):.3f}")
    print(f"IoU >= 0.5    : {100*np.mean(np.array(ious)>=0.5):.1f}%")
    print(f"IoU >= 0.3    : {100*np.mean(np.array(ious)>=0.3):.1f}%")
if centroid_dists:
    print(f"Mean center dist (px)  : {np.mean(centroid_dists):.1f}")
    print(f"Median center dist (px): {np.median(centroid_dists):.1f}")