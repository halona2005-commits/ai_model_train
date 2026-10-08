from pathlib import Path
import json

ROOT = Path(r"D:\internship_RIED\ai-model")
LABELS = ROOT / "DogFLW" / "train" / "labels"

label_files = list(LABELS.glob("*"))
print("Total label files:", len(label_files))
print("Extensions:", set(p.suffix for p in label_files))

print()
print("=" * 60)
print("FIRST 3 LABEL FILE CONTENTS")
print("=" * 60)

for p in label_files[:3]:
    print()
    print("FILE:", p.name)
    text = p.read_text(errors="ignore")
    print(text[:2000])

print()
print("=" * 60)
print("IF JSON: show structure")
print("=" * 60)

json_files = [p for p in label_files if p.suffix.lower() == ".json"]
if json_files:
    with open(json_files[0]) as f:
        data = json.load(f)
    print("Top-level type:", type(data))
    if isinstance(data, dict):
        print("Keys:", list(data.keys()))
        for k, v in data.items():
            print(f"  {k}: type={type(v).__name__}", end="")
            if isinstance(v, list):
                print(f"  len={len(v)}")
                if v and isinstance(v[0], dict):
                    print("    sample item keys:", list(v[0].keys()))
                    print("    sample item:", v[0])
            else:
                print(f"  value={v}")