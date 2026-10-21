# AI Model — Canine Nose-Print Biometric Recognition

A deep learning pipeline designed to identify dogs using their unique nose-print patterns (similar to fingerprint recognition). This repository contains the complete AI pipeline: nose localization, validation, and biometric identity embedding.

## 🏗️ Pipeline Architecture

1. **Nose Localization (YOLOv11n)**: Detects and tightly crops the nose region from a full dog image.
2. **Nose Validator (ResNet50)**: A binary classifier that rejects invalid images (eyes, ears, random objects) with **99.84% accuracy**.
3. **Input Gate (`prepare_nose.py`)**: A unified function used by both registration and recognition to ensure only valid nose crops proceed.
4. **Biometric Engine (ResNet50 + ArcFace)**: Converts the nose crop into a **2048-dimensional identity embedding**. Upgraded from Triplet Loss to ArcFace for robust open-set recognition (rejecting unknown dogs).

> **Note:** Real-ESRGAN has been intentionally removed from this pipeline. Enhancement destroys the fine ridges of the nose print, which are the actual biometric features.

## 📊 Dataset

The model is trained on the full **Pet Biometric Challenge 2022** dataset:
*   **20,000** training images.
*   **6,000** unique dog identities.
*   Images are grouped by dog ID and session clusters (using perceptual hashing) to prevent the model from memorizing the same photo.
*   Train / Validation / Test split is identity-disjoint (4,200 train dogs / 900 val dogs / 900 test dogs).

## 📈 Evaluation & Results

*   **Nose Validator**: 99.84% test accuracy (rejects eyes/ears).
*   **ArcFace Biometric Engine**: Trained for 10 epochs. The loss converged successfully (22.7 -> 15.5).
*   *Final Open-Set Evaluation (Test A & Test B) - [Pending Final Run]*

## 🚀 How to Run

### 1. Requirements
```bash
pip install tensorflow ultralytics opencv-python numpy pandas imagehash