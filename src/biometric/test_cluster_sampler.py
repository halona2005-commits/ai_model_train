from pathlib import Path
import pandas as pd
from src.biometric.batch_sampler import make_pk_dataset

ROOT = Path(r"D:\internship_RIED\ai-model")
SPLITS = ROOT / "dataset_corrected" / "splits"

train_df = pd.read_csv(SPLITS / "train.csv")
print(f"Train rows : {len(train_df)}")
print(f"Train dogs : {train_df['dog_id'].nunique()}")

all_dogs = sorted(train_df["dog_id"].unique().tolist())
dog_to_idx = {d: i for i, d in enumerate(all_dogs)}

ds, gen = make_pk_dataset(
    train_df, dog_to_idx,
    P=8, K=4, augment=True,
    batches_per_epoch=20, seed=42,
)

print("\nSampling 20 batches to measure repeat rate...")
for images, labels in ds:
    pass

print("Sampler stats:", gen.stats())
print()
print("Expected repeat_rate: ~0.0 (cluster-aware sampling prevents same-image repeats)")
print("If repeat_rate > 0.05, there's an issue.")