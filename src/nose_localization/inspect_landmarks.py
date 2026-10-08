from pathlib import Path
import cv2
import numpy as np
import json
import random

ROOT = Path(r"D:\internship_RIED\ai-model")
IMG_DIR = ROOT / "DogFLW" / "test" / "images"
LBL_DIR = ROOT / "DogFLW" / "test" / "labels"
OUT = ROOT / "dataset" / "localization" / "test_images" / "landmark_check"
OUT.mkdir(parents=True, exist_ok=True)

# colors for consecutive landmark groups
GROUPS = [
    (0, 14, (0, 0, 255)),     # blue
    (14, 24, (0, 255, 0)),    # green
    (24, 46, (255, 0, 0)),    # red
]

files = sorted(IMG_DIR.glob("*"))
random.seed(7)
sample = random.sample(files, 5)

for p in sample:
    img = cv2.imread(str(p))
    if img is None:
        continue

    lbl = LBL_DIR / (p.stem + ".json")
    if not lbl.exists():
        continue

    with open(lbl) as f:
        data = json.load(f)

    lms = np.array(data["landmarks"], dtype=np.float32)

    for (a, b, color) in GROUPS:
        pts = lms[a:b]
        mask = np.isfinite(pts).all(axis=1)
        pts = pts[mask]
        for (x, y) in pts:
            cv2.circle(img, (int(x), int(y)), 4, color, -1)
        if pts.shape[0] > 0:
            xs, ys = pts[:, 0], pts[:, 1]
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()), int(ys.max())
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # label the groups
    cv2.putText(img, "0-13 BLUE",   (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "14-23 GREEN", (10, 55),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, "24-45 RED",   (10, 85),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    out_path = OUT / f"lm_{p.stem}.jpg"
    cv2.imwrite(str(out_path), img)
    print("saved:", out_path)

print()
print("Open a few of these images. Each group is a different color:")
print("  BLUE   = landmarks 0–13")
print("  GREEN  = landmarks 14–23")
print("  RED    = landmarks 24–45")
print()
print("Tell me which color sits on the actual nose.")