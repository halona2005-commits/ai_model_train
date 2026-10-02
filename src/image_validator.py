import cv2


def validate_image(image_path, min_width=100, min_height=100, blur_threshold=100):
    """
    Validate whether an image is readable, sufficiently large,
    and not too blurry.

    Returns:
        valid (bool)
        message (str)
        blur_score (float or None)
    """

    # 1. Read image
    image = cv2.imread(image_path)

    if image is None:
        return False, "Image could not be read", None

    # 2. Check resolution
    height, width = image.shape[:2]

    if width < min_width or height < min_height:
        return False, f"Image resolution too low: {width}x{height}", None

    # 3. Calculate blur score
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    # 4. Actually check blur
    if blur_score < blur_threshold:
        return False, f"Image is too blurry: {blur_score:.2f}", blur_score

    return True, f"Image quality acceptable: {blur_score:.2f}", blur_score