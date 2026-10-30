from __future__ import annotations

import cv2
import numpy as np


class InvalidNoseImage(ValueError):
    """Raised when an upload is not usable as a close-up nose-print image."""


def validate_nose_image(
    image_bytes: bytes,
    *,
    min_side: int = 128,
    max_aspect_ratio: float = 1.4,
    min_grayscale_stddev: float = 10.0,
) -> None:
    """Reject inputs that cannot be valid, tightly framed nose-print samples.

    This is deliberately a conservative capture-quality gate, not a semantic
    nose detector.  It prevents arbitrary or wide images from being enrolled
    as biometric templates while a trained nose detector is not yet present.
    """

    if not image_bytes:
        raise InvalidNoseImage("The uploaded image is empty.")

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidNoseImage("The uploaded file is not a readable image.")

    height, width = image.shape[:2]
    if min(height, width) < min_side:
        raise InvalidNoseImage(
            "The image is too small. Send a closer, sharper nose photo."
        )

    aspect_ratio = max(width / height, height / width)
    if aspect_ratio > max_aspect_ratio:
        raise InvalidNoseImage(
            "The image is too wide or tall. Crop tightly around only the nose."
        )

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if float(grayscale.std()) < min_grayscale_stddev:
        raise InvalidNoseImage(
            "The image has too little detail. Send a clear, well-lit nose photo."
        )
