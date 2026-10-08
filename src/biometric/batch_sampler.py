from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

from src.biometric.augmentation import augment_train, augment_eval, IMG_SIZE

# -----------------------------------------------------------
# Constants
# -----------------------------------------------------------
SEED = 42
P_DOGS_PER_BATCH = 8
K_IMAGES_PER_DOG = 4

# -----------------------------------------------------------
# Image loading helper
# -----------------------------------------------------------
def load_image(path):
    """Load one image file -> [224,224,3] float32 in [0,255]."""
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    return img

def preprocess(img):
    """Convert [0,255] image to ResNet50-normalized input."""
    return preprocess_input(img)

# -----------------------------------------------------------
# P-K batch generator
# -----------------------------------------------------------
class PKBatchGenerator:
    """
    Yields batches of shape:
        images : [P*K, 224, 224, 3]  float32, ResNet50-preprocessed
        labels : [P*K]               int32 (dog index 0..num_dogs-1)

    Parameters
    ----------
    df : pd.DataFrame with columns ['abs_path', 'dog_id']
    dog_to_idx : dict mapping dog_id -> int class index
    P : number of dogs per batch
    K : images per dog
    augment : bool, apply training augmentations
    seed : int
    """

    def __init__(self, df, dog_to_idx, P=8, K=4, augment=True, seed=42):
        self.df = df.reset_index(drop=True)
        self.dog_to_idx = dog_to_idx
        self.P = P
        self.K = K
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        # Precompute: dog_id -> np.array of image paths
        grouped = self.df.groupby("dog_id")
        self.dogs = list(grouped.groups.keys())
        self.dog_to_paths = {
            dog_id: grouped.get_group(dog_id)["abs_path"].values
            for dog_id in self.dogs
        }

        self.batch_size = P * K
        self.num_dogs = len(self.dogs)

    def __len__(self):
        # roughly one batch per 20 images
        return max(1, len(self.df) // 20)

    def sample_batch(self):
        # 1. sample P dogs
        chosen_dogs = self.rng.choice(self.dogs, size=self.P, replace=False)

        paths = []
        labels = []
        for dog in chosen_dogs:
            dog_paths = self.dog_to_paths[dog]
            # sample K images with replacement
            idx = self.rng.integers(0, len(dog_paths), size=self.K)
            for i in idx:
                paths.append(dog_paths[i])
                labels.append(self.dog_to_idx[dog])

        # 2. load & augment
        imgs = []
        for p, lbl in zip(paths, labels):
            img = load_image(p)
            if self.augment:
                img = augment_train(img)
            else:
                img = augment_eval(img)
            img = preprocess(img)
            imgs.append(img)

        imgs = tf.stack(imgs)                 # [P*K, 224, 224, 3]
        labels = tf.constant(labels, dtype=tf.int32)
        return imgs, labels

# -----------------------------------------------------------
# Wrap generator as tf.data.Dataset
# -----------------------------------------------------------
def make_pk_dataset(df, dog_to_idx, P=8, K=4, augment=True, batches_per_epoch=50, seed=42):
    """
    Returns a tf.data.Dataset that yields (images, labels) batches.
    Since our sampler is Python (not vectorized tf ops), we wrap it.
    """
    gen = PKBatchGenerator(df, dog_to_idx, P=P, K=K, augment=augment, seed=seed)

    output_signature = (
        tf.TensorSpec(shape=(P * K, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(P * K,),           dtype=tf.int32),
    )

    def generator():
        for _ in range(batches_per_epoch):
            yield gen.sample_batch()

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature,
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, gen