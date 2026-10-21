from pathlib import Path
from src.pipeline.prepare_nose import prepare_nose

ROOT = Path(r"D:\internship_RIED\ai-model")
SPLITS = ROOT / "dataset_corrected" / "train"

# Test 1: a real nose
nose = SPLITS / "images" / "x1FVNAVSRniPE0kHchH0IAAAACMAARAD.jpg"
r = prepare_nose(image_path=nose)
print(f"Real nose     -> {r['status']:>10}  reason={r['reason_code']}  score={r['validator_score']:.3f}")

# Test 2: a non-nose (find any image from a different folder)
# use one of the validator negatives
eye = ROOT / "dataset" / "validator" / "raw" / "not_dog_nose" / "dog_eye"
eye_img = next(eye.glob("*.jpg"), None)
if eye_img:
    r = prepare_nose(image_path=eye_img)
    print(f"Dog eye       -> {r['status']:>10}  reason={r['reason_code']}  score={r['validator_score']:.3f}")

# Test 3: missing file
r = prepare_nose(image_path=ROOT / "does_not_exist.jpg")
print(f"Missing file  -> {r['status']:>10}  reason={r['reason_code']}")