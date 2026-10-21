"""
v3: larger batch, longer training.

Everything else same as v2.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import tensorflow as tf

from src.biometric.batch_sampler import make_pk_dataset
from src.biometric.model import build_embedding_model, batch_all_triplet_loss


ROOT = Path(r"D:\internship_RIED\ai-model")
SPLITS = ROOT / "dataset_corrected" / "splits"
MODEL_DIR = ROOT / "models" / "biometric"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

P = 16              # <-- increased from 8
K = 4
MARGIN = 0.3

BATCHES_PER_EPOCH = 40   # roughly same total images per epoch
VAL_BATCHES       = 10

PHASE1_EPOCHS = 8        # <-- increased
PHASE2_EPOCHS = 5

PHASE1_UNFREEZE_LAST = 20
PHASE2_UNFREEZE_LAST = 60

LR_PHASE1 = 1e-4
LR_PHASE2 = 1e-5

SEED = 42


train_df = pd.read_csv(SPLITS / "train.csv")
val_df   = pd.read_csv(SPLITS / "val.csv")

train_dogs = sorted(train_df["dog_id"].unique().tolist())
val_dogs   = sorted(val_df["dog_id"].unique().tolist())

train_dog_to_idx = {d: i for i, d in enumerate(train_dogs)}
val_dog_to_idx   = {d: i for i, d in enumerate(val_dogs)}

print("Train dogs:", len(train_dogs), "images:", len(train_df))
print("Val   dogs:", len(val_dogs),   "images:", len(val_df))


model, backbone = build_embedding_model(unfreeze_last_n=PHASE1_UNFREEZE_LAST)
n_trainable = sum(1 for v in model.trainable_variables)
print(f"Trainable variables: {n_trainable}")
if n_trainable == 0:
    raise RuntimeError("No trainable variables")


train_loss_tracker   = tf.keras.metrics.Mean(name="loss")
train_active_tracker = tf.keras.metrics.Mean(name="active")
val_loss_tracker     = tf.keras.metrics.Mean(name="val_loss")

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)


def make_train_step(optimizer):
    @tf.function
    def step(images, labels):
        with tf.GradientTape() as tape:
            emb = model(images, training=True)
            loss, frac = batch_all_triplet_loss(labels, emb, margin=MARGIN)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, frac
    return step


@tf.function
def val_step(images, labels):
    emb = model(images, training=False)
    loss, _ = batch_all_triplet_loss(labels, emb, margin=MARGIN)
    return loss


def run_epoch(train_step_fn, train_ds, epoch, phase):
    train_loss_tracker.reset_state()
    train_active_tracker.reset_state()
    t0 = time.time()
    n = 0
    for images, labels in train_ds:
        loss, frac = train_step_fn(images, labels)
        train_loss_tracker.update_state(loss)
        train_active_tracker.update_state(frac)
        n += 1
        if n % 10 == 0:
            log(f"  [{phase}] epoch {epoch}  step {n}/{BATCHES_PER_EPOCH}  "
                f"loss={train_loss_tracker.result():.4f}  "
                f"active={train_active_tracker.result():.3f}")
    return train_loss_tracker.result().numpy(), time.time() - t0


def run_val(val_ds):
    val_loss_tracker.reset_state()
    for images, labels in val_ds:
        loss = val_step(images, labels)
        val_loss_tracker.update_state(loss)
    return val_loss_tracker.result().numpy()


best_val = float("inf")
best_path = MODEL_DIR / "best_biometric_v3.keras"

log("=" * 60)
log(f"PHASE 1: unfreeze last {PHASE1_UNFREEZE_LAST}, LR={LR_PHASE1}, batch={P*K}")
log("=" * 60)

optimizer = tf.keras.optimizers.Adam(LR_PHASE1)
train_step_fn = make_train_step(optimizer)

for epoch in range(1, PHASE1_EPOCHS + 1):
    train_ds, _ = make_pk_dataset(train_df, train_dog_to_idx, P=P, K=K,
                                  augment=True, batches_per_epoch=BATCHES_PER_EPOCH,
                                  seed=SEED + epoch)
    val_ds, _ = make_pk_dataset(val_df, val_dog_to_idx, P=P, K=K,
                                augment=False, batches_per_epoch=VAL_BATCHES,
                                seed=SEED + 1000 + epoch)

    train_loss, dt = run_epoch(train_step_fn, train_ds, epoch, "P1")
    val_loss = run_val(val_ds)
    log(f"Epoch {epoch:02d}  train_loss={train_loss:.4f}  "
        f"val_loss={val_loss:.4f}  time={dt:.1f}s")
    if val_loss < best_val:
        best_val = val_loss
        model.save(best_path)
        log(f"  new best val_loss={best_val:.4f}  saved")


log("")
log("=" * 60)
log(f"PHASE 2: unfreeze last {PHASE2_UNFREEZE_LAST}, LR={LR_PHASE2}")
log("=" * 60)

backbone.trainable = True
for layer in backbone.layers[:-PHASE2_UNFREEZE_LAST]:
    layer.trainable = False

optimizer = tf.keras.optimizers.Adam(LR_PHASE2)
train_step_fn = make_train_step(optimizer)

for epoch in range(1, PHASE2_EPOCHS + 1):
    train_ds, _ = make_pk_dataset(train_df, train_dog_to_idx, P=P, K=K,
                                  augment=True, batches_per_epoch=BATCHES_PER_EPOCH,
                                  seed=SEED + 5000 + epoch)
    val_ds, _ = make_pk_dataset(val_df, val_dog_to_idx, P=P, K=K,
                                augment=False, batches_per_epoch=VAL_BATCHES,
                                seed=SEED + 6000 + epoch)

    train_loss, dt = run_epoch(train_step_fn, train_ds, epoch, "P2")
    val_loss = run_val(val_ds)
    log(f"Epoch {epoch:02d}  train_loss={train_loss:.4f}  "
        f"val_loss={val_loss:.4f}  time={dt:.1f}s")
    if val_loss < best_val:
        best_val = val_loss
        model.save(best_path)
        log(f"  new best val_loss={best_val:.4f}  saved")


log("")
log(f"Training done. Best val_loss: {best_val}")
log(f"Model: {best_path}")

(MODEL_DIR / "training_log_v3.txt").write_text("\n".join(log_lines))

print()
print("Best model :", best_path)
print("Log        :", MODEL_DIR / "training_log_v3.txt")