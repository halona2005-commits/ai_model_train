import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input


def preprocess_nose_image(image_path):
    """
    Preprocess a Pet Biometric nose-print image
    before ResNet50 feature extraction.
    """

    # 1. Read image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # 2. Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 3. Resize to ResNet50 input size
    image = cv2.resize(image, (224, 224))

    # 4. Convert to float32
    image = image.astype(np.float32)

    # 5. ResNet50 preprocessing
    image = preprocess_input(image)

    # 6. Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image