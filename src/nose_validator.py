import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


# ============================================================
# PATH
# ============================================================

DATASET_DIR = (
    r"E:\canine-nose-print-recognition\ai-model"
    r"\dataset\nose_validation"
)

MODEL_DIR = (
    r"E:\canine-nose-print-recognition\ai-model"
    r"\models"
)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 10
SEED = 42


# ============================================================
# LOAD DATASET
# ============================================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)


# ============================================================
# CHECK CLASSES
# ============================================================

print("\nClass names:")
print(train_dataset.class_names)


# ============================================================
# PREPROCESSING
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE


def preprocess_dataset(image, label):

    image = preprocess_input(image)

    return image, label


train_dataset = train_dataset.map(
    preprocess_dataset,
    num_parallel_calls=AUTOTUNE
)

validation_dataset = validation_dataset.map(
    preprocess_dataset,
    num_parallel_calls=AUTOTUNE
)


# ============================================================
# PERFORMANCE
# ============================================================

train_dataset = train_dataset.prefetch(
    AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    AUTOTUNE
)


# ============================================================
# RESNET50
# ============================================================

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)


# Freeze ResNet50
base_model.trainable = False


# ============================================================
# CLASSIFIER
# ============================================================

model = models.Sequential([

    layers.Input(
        shape=(224, 224, 3)
    ),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.3),

    layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=[
        "accuracy",

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# TRAIN
# ============================================================

print("\n====================================")
print("STARTING TRAINING")
print("====================================")
print()

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "dog_nose_validator.keras"
)

model.save(
    MODEL_PATH
)


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("====================================")
print("TRAINING COMPLETE")
print("====================================")

print(
    f"Model saved to:\n{MODEL_PATH}"
)

print()

print(
    f"Final training accuracy: "
    f"{history.history['accuracy'][-1]:.4f}"
)

print(
    f"Final validation accuracy: "
    f"{history.history['val_accuracy'][-1]:.4f}"
)

print(
    f"Final validation precision: "
    f"{history.history['val_precision'][-1]:.4f}"
)

print(
    f"Final validation recall: "
    f"{history.history['val_recall'][-1]:.4f}"
)