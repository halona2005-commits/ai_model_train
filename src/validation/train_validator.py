import os
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.utils.class_weight import compute_class_weight

# ---------------- PATHS ----------------
ROOT    = Path(r"D:\internship_RIED\ai-model")
VAL_DIR = ROOT / "dataset" / "validator"
SPLITS  = VAL_DIR / "splits"
MODEL_DIR = ROOT / "models" / "nose_validator"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH    = 32
AUTOTUNE = tf.data.AUTOTUNE
SEED     = 42

# ---------------- DATA ----------------
def load_split(name):
    df = pd.read_csv(SPLITS / f"{name}.csv")
    # absolute path
    df["abs_path"] = df["image_path"].apply(lambda p: str(ROOT / p))
    # verify
    missing = (~df["abs_path"].apply(lambda p: Path(p).exists())).sum()
    print(f"{name}: rows={len(df)}  missing_files={missing}")
    return df

train_df = load_split("train")
val_df   = load_split("val")
test_df  = load_split("test")

# ---------------- tf.data ----------------
def decode_img(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img, tf.cast(label, tf.float32)

def make_ds(df, training=False):
    paths  = df["abs_path"].values
    labels = df["label"].values.astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(decode_img, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(AUTOTUNE)
    return ds

train_ds = make_ds(train_df, training=True)
val_ds   = make_ds(val_df)
test_ds  = make_ds(test_df)

# ---------------- CLASS WEIGHTS ----------------
y = train_df["label"].values
classes = np.unique(y)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
print("Class weights:", class_weight)

# ---------------- MODEL ----------------
def build_model():
    base = ResNet50(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    base.trainable = False

    inp = Input(shape=(224, 224, 3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inp, out)
    return model, base

model, base = build_model()

# ---------------- PHASE 1: frozen backbone ----------------
print("\n" + "=" * 60)
print("PHASE 1: training head (backbone frozen)")
print("=" * 60)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / "best_phase1.keras"),
        monitor="val_auc", mode="max", save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=4, restore_best_weights=True
    ),
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8,
    class_weight=class_weight,
    callbacks=callbacks,
)

# ---------------- PHASE 2: fine-tune last blocks ----------------
print("\n" + "=" * 60)
print("PHASE 2: fine-tune last ResNet50 blocks")
print("=" * 60)

base.trainable = True
# freeze all but last ~30 layers
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

callbacks2 = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / "nose_validator_resnet50.keras"),
        monitor="val_auc", mode="max", save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=4, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc", mode="max", factor=0.5, patience=2, min_lr=1e-7
    ),
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weight,
    callbacks=callbacks2,
)

# ---------------- FINAL EVAL ----------------
print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

results = model.evaluate(test_ds, return_dict=True)
for k, v in results.items():
    print(f"  {k}: {v:.4f}")

# ---------------- PER-TYPE ANALYSIS ----------------
print("\n" + "=" * 60)
print("PER-NEGATIVE-TYPE ANALYSIS (must be LOW confidence)")
print("=" * 60)

test_paths = test_df["abs_path"].values
test_labels = test_df["label"].values
test_types = test_df["negative_type"].values

def predict_paths(paths, batch=32):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(lambda p: decode_img(p, 0)[0], num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch).prefetch(AUTOTUNE)
    return model.predict(ds, verbose=0).ravel()

preds = predict_paths(test_paths)

print()
print(f"{'type':<12} {'n':>6} {'mean_p':>9} {'min':>8} {'max':>8}")
print("-" * 50)
for t in sorted(set(test_types)):
    mask = test_types == t
    p = preds[mask]
    print(f"{t:<12} {len(p):>6} {p.mean():>9.4f} {p.min():>8.4f} {p.max():>8.4f}")

# ---------------- CRITICAL CHECKS ----------------
print("\n" + "=" * 60)
print("CRITICAL CHECKS")
print("=" * 60)

# 1. dog_nose should be HIGH
nose_mask = test_types == "dog_nose"
nose_preds = preds[nose_mask]
nose_high = (nose_preds >= 0.5).mean()
print(f"dog_nose with p>=0.5 (should be HIGH): {nose_high*100:.2f}%")

# 2. Each negative should be LOW
for t in ["dog_eye", "dog_mouth", "dog_face"]:
    m = test_types == t
    p = preds[m]
    low = (p < 0.5).mean()
    print(f"{t} with p<0.5 (should be HIGH):     {low*100:.2f}%")

# ---------------- MANUAL TEST SUGGESTIONS ----------------
print("\n" + "=" * 60)
print("MANUAL CONFUSION CHECK")
print("=" * 60)
print("Pick a few eye/face/mouth images and confirm they get LOW scores.")
print("Sample paths:")
for t in ["dog_eye", "dog_mouth", "dog_face"]:
    m = np.where(test_types == t)[0]
    if len(m):
        i = m[0]
        print(f"  {t}: {test_paths[i]}  ->  p={preds[i]:.4f}")

print("\nModel saved:")
print(" ", MODEL_DIR / "nose_validator_resnet50.keras")