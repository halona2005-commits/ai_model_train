import tensorflow as tf

IMG_SIZE = (224, 224)

# -----------------------------------------------------------
# Training augmentation pipeline
# -----------------------------------------------------------
# Applied to a [H, W, 3] float32 tensor in range [0, 255].
# Returns augmented tensor of the same shape.
# -----------------------------------------------------------

def augment_train(img):
    img = tf.cast(img, tf.float32)

    # Random rotation (±15°)
    img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32)) if False else img
    # (rot90 disabled - we use RandomRotation via Keras preprocessing instead)

    # Brightness
    img = tf.image.random_brightness(img, max_delta=0.20 * 255.0)

    # Contrast
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)

    # Small saturation tweak (very mild, mostly preserved)
    img = tf.image.random_saturation(img, lower=0.9, upper=1.1)

    # Random crop-zoom-translate via affine-ish
    # We use a small crop + resize to simulate zoom + translation
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    scale = tf.random.uniform([], 0.90, 1.0)
    crop_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
    crop_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
    # random offset
    off_h = tf.random.uniform([], 0, h - crop_h + 1, dtype=tf.int32)
    off_w = tf.random.uniform([], 0, w - crop_w + 1, dtype=tf.int32)
    img = tf.image.crop_to_bounding_box(img, off_h, off_w, crop_h, crop_w)
    img = tf.image.resize(img, IMG_SIZE)

    # Occasional mild blur
    if tf.random.uniform([]) < 0.15:
        img = tf.image.resize(
            tf.image.resize(img, (IMG_SIZE[0] // 2, IMG_SIZE[1] // 2)),
            IMG_SIZE
        )

    # Clip to valid range
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img


def augment_eval(img):
    """No augmentation for validation / test."""
    return tf.cast(img, tf.float32)