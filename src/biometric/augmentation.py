"""
Training augmentation for nose-print biometrics.

Version-safe implementation:
    - Uses explicit tf.expand_dims for batch dimension (Keras 3 requirement)
    - Manual brightness/contrast (avoids version-specific layer behavior)
    - ±15° rotation (real)
    - No 90°/180° rotation
    - No vertical/horizontal flip
    - No hue shift

Input: [H, W, 3] float32 in [0, 255]
Output: [H, W, 3] float32 in [0, 255]
"""

import tensorflow as tf
import math

IMG_SIZE = (224, 224)


# -----------------------------------------------------------------
# Cached Keras preprocessing layers
# Each is applied to a batched tensor: (B, H, W, C)
# -----------------------------------------------------------------
_rotate = tf.keras.layers.RandomRotation(
    factor=15.0 / 360.0,      # ±15°
    fill_mode="reflect",
)
_translate = tf.keras.layers.RandomTranslation(
    height_factor=0.05,
    width_factor=0.05,
    fill_mode="reflect",
)
_zoom = tf.keras.layers.RandomZoom(
    height_factor=0.10,
    width_factor=0.10,
    fill_mode="reflect",
)


def augment_train(img):
    """
    Apply training augmentation to a single nose image.

    img: [H, W, 3] float32 in [0, 255]
    returns: [H, W, 3] float32 in [0, 255]
    """
    img = tf.cast(img, tf.float32)

    # add batch dim -> (1, H, W, 3), Keras 3 requires this
    x = tf.expand_dims(img, axis=0)

    # rotation ±15°
    x = _rotate(x, training=True)

    # translation ±5%
    x = _translate(x, training=True)

    # zoom ±10%
    x = _zoom(x, training=True)

    # remove batch dim
    x = x[0]

    # brightness ±20%  (manual, range-safe on [0, 255])
    delta = tf.random.uniform([], -0.20, 0.20) * 255.0
    x = x + delta

    # contrast ±15%  (manual, mean-centered multiply)
    mean = tf.reduce_mean(x, axis=[0, 1], keepdims=True)
    factor = tf.random.uniform([], 0.85, 1.15)
    x = (x - mean) * factor + mean

    # clamp to valid range
    x = tf.clip_by_value(x, 0.0, 255.0)

    return x


def augment_eval(img):
    """Validation / test: no augmentation, just type-safe cast."""
    return tf.cast(img, tf.float32)