# AI Model — Canine Nose-Print Recognition

A Python-based canine nose-print recognition module designed to process dog images, detect and extract canine nose regions, enhance the extracted nose images, and generate 2048-dimensional feature vectors for biometric identification.

## Features

### Dual Dog Image Input

The pipeline supports two types of input:

- **Full Dog Image**
  - Detects the dog's face.
  - Crops the detected face.
  - Detects the nose inside the face crop.
  - Extracts the nose region.

- **Dog Nose Image**
  - Directly accepts an already-cropped dog nose image.
  - Skips face detection and nose detection.

Both input types continue through nose enhancement and feature extraction.

### Image Quality Check

The pipeline first checks whether the uploaded image satisfies the minimum requirements for processing.

Current prototype checks include:

- Image readability
- Minimum image dimensions
- Image sharpness using Laplacian variance

The current minimum sharpness threshold is:

```text
50
```

This threshold is a prototype configuration and can be calibrated further using real deployment data.

### Dog Input Validation & Routing

Dog validation and input routing are performed at Step 2 of the pipeline.

The system first attempts to detect a dog face.

If a dog face is detected, the image is treated as a full dog image.

If no dog face is detected, the trained dog-nose validator is used to determine whether the uploaded image is already a dog nose image.

The dog-nose validator is used only for input routing and is **not applied after nose cropping**.

### Dog Face Detection

A YOLO11n-based dog face detector was trained using the DogFLW dataset.

Dataset:

```text
Total samples:       3,855
Training samples:    3,084
Validation samples:    771
Classes:             dog_face
```

The final face annotations were created from DogFLW facial landmarks with an additional margin around the detected facial region.

The annotation dataset was audited and repaired before training.

The final detector was tested on 20 random dog images:

```text
Successful detections: 20/20
```

Model:

```text
models/face_detector/yolo11n_face_final/weights/best.pt
```

### Dog Nose Detection

A YOLO11n-based nose detector identifies the canine nose inside the detected dog face.

Two trained model weights are used during runtime:

```text
models/nose_detector/yolo11n_nose_tight/weights/best.pt

models/nose_detector/yolo11n_nose_augmented/weights/last.pt
```

The runtime uses the primary nose detector first and falls back to the second trained weight when the primary detector does not detect a nose.

The nose detector was trained using DogFLW-derived nose annotations with additional angled and side-view canine nose images.

### Nose Crop Extraction

After detecting the nose, a dedicated cropper extracts the nose region from the face crop.

The cropping process includes:

- YOLO bounding-box detection
- Central-region refinement
- Dark-region and contour analysis
- Bounding-box constraints
- Center-shift validation
- Padding around the detected region
- Square crop generation

Runtime testing successfully produced nose crops for:

```text
19/20 test images
```

### Dog Nose Validator

A ResNet50-based binary classifier is used to determine whether an image without a detected dog face is a dog nose image.

Validation dataset:

```text
Positive samples: 300 dog nose images
Negative samples: 300 non-dog-nose images
```

Negative samples included:

- Humans: 60
- Other animals: 120
- Random objects: 120

Dataset split:

```text
Training:    480
Validation:  120
```

Validation results:

```text
Validation Accuracy: 99.17%
Precision:           100.00%
Recall:               98.31%
Validation Loss:       0.0303
```

An additional external test containing 8 images produced:

```text
Correct classifications: 8/8
```

Model:

```text
models/dog_nose_validator.keras
```

The saved model does not contain an internal preprocessing layer. ImageNet `preprocess_input` is therefore applied externally during inference.

### Nose Enhancement

The extracted nose crop is enhanced before feature extraction.

The enhancement pipeline uses:

- Real-ESRGAN
- RRDBNet x4plus
- 4x super-resolution
- Brightness correction
- Richardson-Lucy deblurring
- Image sharpening
- Lanczos resizing

The final image is resized to:

```text
224 x 224
```

Runtime configuration uses tiled Real-ESRGAN processing:

```text
Tile size: 128
Tile padding: 10
Scale: 4
```

All 19 successfully cropped runtime nose images were processed successfully.

### ResNet50 Feature Extraction

The enhanced canine nose image is processed using an ImageNet-pretrained ResNet50 model.

Processing configuration:

```text
Model:        ResNet50
Weights:      ImageNet
include_top:  False
Pooling:      Global Average Pooling
Output:       2048 dimensions
```

The input image is:

```text
RGB
224 x 224
ImageNet preprocessed
```

The resulting feature vector has the shape:

```text
(1, 2048)
```

Runtime verification:

```text
Images processed: 19
Feature vectors:  19
Feature shape:    (19, 2048)
NaN values:       0
Inf values:       0
Failed images:    0
```

### Canine Biometric Feature Database

The clean biometric dataset currently contains:

```text
Unique images:  2,892
Unique Dog IDs: 1,064
```

Precomputed ResNet50 features are stored as:

```text
features/features.npy
features/dog_ids.npy
features/image_names.npy
```

Runtime-generated features are stored separately:

```text
features/runtime/runtime_features.npy
features/runtime/runtime_image_names.npy
```

The feature database provides the representation required for the later dog-matching stage.

---

## Installation

### Prerequisites

- Python 3.10 or 3.11
- TensorFlow
- OpenCV
- NumPy
- Pillow
- scikit-learn
- Ultralytics YOLO
- Real-ESRGAN
- PyTorch
- ResNet50

### Setup

```bash
# Clone the repository
git clone https://github.com/bhavani-patil04/canine-nose-print-recognition.git

# Navigate to the AI model
cd canine-nose-print-recognition/ai-model

# Create a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start & CLI Usage

### 1. Process a Full Dog Image

```powershell
python src/pipeline/run_pipeline.py "random_dog\dog1.jpg"
```

The pipeline processes the image through:

```text
Image Quality Check
Dog Face Detection
Face Crop
Nose Detection
Nose Crop
Nose Enhancement
ResNet50 Feature Extraction
2048-D Feature Vector
```

### 2. Process a Dog Nose Image

```powershell
python src/pipeline/run_pipeline.py "path\to\nose_image.jpg"
```

For a nose-only input, the image proceeds directly through nose validation, enhancement, and feature extraction.

No face crop or nose crop is performed.

---

## Output

For a full dog image, the pipeline produces:

```text
pipeline_results/
└── <input_name>/
    ├── face_crop.jpg
    ├── nose_crop.jpg
    ├── nose_enhanced.png
    └── feature_2048.npy
```

For a dog nose input:

```text
pipeline_results/
└── <input_name>/
    ├── nose_input.jpg
    ├── nose_enhanced.png
    └── feature_2048.npy
```

The final feature file contains the extracted:

```text
2048-D feature vector
```

---

## Model Summary

| Component | Model / Method | Purpose |
|---|---|---|
| Face Detection | YOLO11n | Detect dog face |
| Nose Detection | YOLO11n | Detect dog nose |
| Nose Validation | ResNet50 | Validate nose-only input |
| Enhancement | Real-ESRGAN x4plus | Improve nose image quality |
| Feature Extraction | ResNet50 | Generate 2048-D representation |
| Feature Matching | Cosine Similarity | Separate dog identification stage |

---

## Validation Summary

### Dog Face Detection

```text
Test images:          20
Successful detection: 20/20
```

### Nose Cropping

```text
Successful crops: 19/20
```

### Nose Enhancement

```text
Successful enhancement: 19/19
```

### Feature Extraction

```text
Images:       19
Shape:        (19, 2048)
NaN values:   0
Inf values:   0
Failures:     0
```

### Dog Nose Validation

```text
Validation Accuracy: 99.17%
Precision:           100.00%
Recall:               98.31%

External test:
8/8 correct
```

> Note: The 8-image external test is a small sanity check and should not be interpreted as a generalization benchmark. The 100-image evaluation performed during development used images from the same 600-image validation dataset and therefore was not an independent test set.

---

## Project Structure

```text
ai-model/
|
├── dataset/
│   ├── DogFLW/
│   ├── nose_validation/
│   ├── face_detection_yolo/
│   ├── pet_biometric_clean/
│   └── ...
|
├── features/
│   ├── features.npy
│   ├── dog_ids.npy
│   ├── image_names.npy
│   └── runtime/
│       ├── runtime_features.npy
│       └── runtime_image_names.npy
|
├── models/
│   ├── face_detector/
│   │   └── yolo11n_face_final/
│   │
│   ├── nose_detector/
│   │   ├── yolo11n_nose_tight/
│   │   └── yolo11n_nose_augmented/
│   │
│   ├── dog_nose_validator.keras
│   └── RealESRGAN_x4plus.pth
|
├── src/
│   ├── pipeline/
│   │   └── run_pipeline.py
│   │
│   ├── face_detection/
│   ├── feature_extraction/
│   ├── nose_localization/
│   ├── preprocessing.py
│   ├── feature_extractor.py
│   ├── extract_all_features.py
│   ├── nose_validator.py
│   └── ...
|
├── requirements.txt
└── README.md
```

---

## Development Notes

- DogFLW was used for dog face and nose localization development.
- The canine biometric dataset was cleaned before feature extraction.
- The final clean biometric dataset contains 2,892 unique images representing 1,064 Dog IDs.
- The face detector annotations were audited and repaired using DogFLW landmarks.
- Two nose detector weights are retained to provide runtime fallback.
- Older nose-localization experiments are retained as development history.
- The dog-nose validator is used only for input routing.
- Nose validation is not repeated after a nose crop has already been produced.
- The current image-quality threshold is a prototype value and requires calibration using deployment data.
- The current pipeline produces feature vectors.
- Overall dog identification accuracy has not yet been established.

## License

Private repository. All rights reserved.