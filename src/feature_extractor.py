import numpy as np
from tensorflow.keras.applications import ResNet50


# Load pretrained ResNet50 as a feature extractor
model = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)


def extract_features(preprocessed_image):
    """
    Extract a 2048-dimensional feature vector
    from a preprocessed nose image.
    """

    features = model.predict(preprocessed_image, verbose=0)

    return features