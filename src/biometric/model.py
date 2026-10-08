import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import ResNet50


EMBED_DIM = 2048
IMG_SIZE = (224, 224)


def build_embedding_model(unfreeze_last_n=20):
    """
    Build ResNet50 -> GAP -> L2-normalized 2048-D embedding.
    Only the last `unfreeze_last_n` layers of ResNet50 are trainable.
    """
    backbone = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="avg",                 # GAP -> 2048-D
    )

    # Freeze all, then unfreeze last N
    backbone.trainable = True
    for layer in backbone.layers[:-unfreeze_last_n]:
        layer.trainable = False

    inp = Input(shape=(224, 224, 3))
    x = backbone(inp, training=False)
    x = layers.UnitNormalization(axis=1, name="l2_norm")(x)

    model = Model(inp, x, name="nose_embedding")
    return model, backbone


def set_trainable_last_n(backbone, n):
    """Change how many trailing layers of the backbone are trainable."""
    backbone.trainable = True
    for layer in backbone.layers[:-n]:
        layer.trainable = False
    # print a summary of how many params are now trainable
    total = 0
    for layer in backbone.layers:
        if layer.trainable:
            total += layer.count_params()
    print(f"  [backbone] trainable layers: last {n}  |  "
          f"trainable params in backbone: {total:,}")


def batch_all_triplet_loss(labels, embeddings, margin=0.3):
    """
    Batch-all triplet loss with hardest positive and hardest negative.
    labels     : [B] int32
    embeddings : [B, 2048] float32 (already L2-normalized)
    returns (loss_scalar, fraction_active_triplets)
    """
    dot = tf.matmul(embeddings, embeddings, transpose_b=True)
    square_norm = tf.linalg.diag_part(dot)
    distances = (
        tf.expand_dims(square_norm, 0)
        - 2.0 * dot
        + tf.expand_dims(square_norm, 1)
    )
    distances = tf.maximum(distances, 0.0)

    labels = tf.cast(labels, tf.int32)
    labels_eq = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    labels_neq = tf.logical_not(labels_eq)
    eye = tf.eye(tf.shape(labels)[0], dtype=tf.bool)
    pos_mask = tf.logical_and(labels_eq, tf.logical_not(eye))
    neg_mask = labels_neq

    pos_dist = tf.where(pos_mask, distances, tf.zeros_like(distances))
    hardest_pos = tf.reduce_max(pos_dist, axis=1)

    big = tf.fill(tf.shape(distances), tf.constant(1e9, dtype=distances.dtype))
    neg_dist = tf.where(neg_mask, distances, big)
    hardest_neg = tf.reduce_min(neg_dist, axis=1)

    loss_vec = tf.maximum(hardest_pos - hardest_neg + margin, 0.0)

    has_pos = tf.reduce_any(pos_mask, axis=1)
    has_neg = tf.reduce_any(neg_mask, axis=1)
    valid = tf.logical_and(has_pos, has_neg)
    valid_f = tf.cast(valid, loss_vec.dtype)

    loss = tf.reduce_sum(loss_vec * valid_f) / (tf.reduce_sum(valid_f) + 1e-9)
    frac_active = tf.reduce_mean(
        tf.cast(tf.greater(loss_vec, 0.0), tf.float32) * valid_f
    )
    return loss, frac_active