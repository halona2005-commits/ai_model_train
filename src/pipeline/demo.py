from pathlib import Path
import pandas as pd
from src.pipeline.recognize import recognize

ROOT = Path(r"D:\internship_RIED\ai-model")
SPLITS = ROOT / "dataset" / "biometric" / "splits"


def show_result(case_name, image_path, true_dog_id, result):
    print()
    print("=" * 62)
    print(f"  {case_name}")
    print("=" * 62)
    print(f"  Image     : {Path(image_path).name}")
    print(f"  True ID   : {true_dog_id}")
    print(f"  YOLO used : {result.get('used_yolo')}")
    print(f"  Validator : {result.get('validator_score', 0):.4f}")
    print("  " + "-" * 58)

    status = result["status"]
    if status == "matched":
        print(f"  STATUS    : MATCHED")
        print(f"  Dog ID    : {result['dog_id']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        if true_dog_id is not None and result["dog_id"] == true_dog_id:
            print(f"  Verdict   : CORRECT")
        elif true_dog_id is not None:
            print(f"  Verdict   : WRONG (expected {true_dog_id})")
    elif status == "unknown":
        print(f"  STATUS    : UNKNOWN")
        print(f"  Best guess: dog_id={result['best_guess']['dog_id']}  "
              f"sim={result['best_guess']['similarity']:.4f}")
    elif status == "not_a_nose":
        print(f"  STATUS    : NOT A NOSE")
        print(f"  Reason    : {result['message']}")
    else:
        print(f"  STATUS    : ERROR")
        print(f"  Message   : {result.get('message')}")

    if "top5" in result:
        print("  " + "-" * 58)
        print("  Top-5:")
        for i, m in enumerate(result["top5"], 1):
            mark = ""
            if true_dog_id is not None and m["dog_id"] == true_dog_id:
                mark = "  <-- true dog"
            print(f"    #{i}  dog_id={m['dog_id']:<5}  sim={m['similarity']:.4f}{mark}")


train_df = pd.read_csv(SPLITS / "train.csv")
test_df  = pd.read_csv(SPLITS / "test.csv")

print("=" * 62)
print("  CANINE NOSE-PRINT RECOGNITION — END-TO-END DEMO")
print("=" * 62)
print(f"  Database : {len(train_df)} train images")
print(f"  Queries  : test split ({test_df['dog_id'].nunique()} unseen dogs)")
print()
print("  Loading models (first call ~30s)...")

# Case 1: registered
row = train_df.iloc[0]
show_result("CASE 1 — Registered dog",
            row["abs_path"], int(row["dog_id"]), recognize(row["abs_path"]))

# Case 2: unseen
row = test_df.iloc[0]
show_result("CASE 2 — Unseen dog",
            row["abs_path"], int(row["dog_id"]), recognize(row["abs_path"]))

# Case 3: another unseen
row = test_df.iloc[10]
show_result("CASE 3 — Another unseen dog",
            row["abs_path"], int(row["dog_id"]), recognize(row["abs_path"]))

# Case 4: full dog photo (YOLO should detect)
yolo_sample = ROOT / "DogFLW" / "test" / "images" / "n02090379_91.png"
if yolo_sample.exists():
    show_result("CASE 4 — Full dog photo (YOLO)",
                yolo_sample, None, recognize(yolo_sample))

# Case 5: not a nose
not_nose = ROOT / "DogFLW" / "test" / "images" / "n02110806_3974.png"
if not_nose.exists():
    show_result("CASE 5 — Random image",
                not_nose, None, recognize(not_nose))


print()
print("=" * 62)
print("  SUMMARY")
print("=" * 62)
print("  MATCHED     -> dog registered, dog_id returned")
print("  UNKNOWN     -> not in database, prompt to register")
print("  NOT_A_NOSE  -> input is not a valid nose")
print()
print("  Front-end calls:  recognize(image_path)")
print("=" * 62)