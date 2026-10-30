import tensorflow as tf
import math
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable()
class ArcFaceLayer(layers.Layer):
    def __init__(self, num_classes, margin=0.5, scale=30.0, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.cos_m = tf.math.cos(margin)
        self.sin_m = tf.math.sin(margin)
        self.th = tf.math.cos(math.pi - margin)
        self.mm = tf.math.sin(math.pi - margin) * margin

    def build(self, input_shape):
        self.W = self.add_weight(
            name="arcface_weights",
            shape=(input_shape[0][-1], self.num_classes),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs):
        embeddings, labels = inputs
        W_norm = tf.math.l2_normalize(self.W, axis=0)
        cosine = tf.matmul(embeddings, W_norm)
        sine = tf.sqrt(tf.maximum(1.0 - tf.square(cosine), 0.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = tf.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = tf.one_hot(tf.cast(labels, tf.int32), self.num_classes)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.scale