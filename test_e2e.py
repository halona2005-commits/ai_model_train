import cv2
from src.pipeline.recognize import recognize

# Using a test image from the dataset
img_path = r"D:\internship_RIED\ai-model\pet_biometric_full\train\images\--1WCesjS6CGTNbuCcv8NwAAACMAARAD.jpg"

print(f"Testing Raw Image: {img_path}")
print("Running YOLO -> Validator -> ArcFace -> Match...")

# This single call runs the entire pipeline
result = recognize(img_path)

print("\n--- FINAL API RESULT ---")
print(result)