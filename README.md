# ArtifactLens — Model Training & Inference

> **The ML backbone of ArtifactLens** — ResNet50-based deepfake detection model trained on 138K+ face images across 20 epochs on an NVIDIA RTX A6000. Version 2 achieves **82.99% validation accuracy** through class-weighted fine-tuning.

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![ResNet50](https://img.shields.io/badge/Architecture-ResNet50-blueviolet?style=flat)
![MTCNN](https://img.shields.io/badge/Face%20Detection-MTCNN-0097A7?style=flat)
![GPU](https://img.shields.io/badge/Hardware-NVIDIA%20RTX%20A6000-76B900?style=flat&logo=nvidia&logoColor=white)
![Accuracy](https://img.shields.io/badge/Model%20Accuracy-82.99%25-success?style=flat)

> 🔗 This repository contains the training pipeline and inference module for the [ArtifactLens](https://github.com/Corerishi/ArtifactLens) full-stack deepfake detection web app.

---

## What is this?

This repo documents the complete **model development pipeline** for ArtifactLens — from dataset preparation and preprocessing (Phase 1) to model training, checkpointing, and inference (Phase 2).

The model is a **fine-tuned ResNet50** classifier trained to distinguish real vs. AI-generated (deepfake) face images. Version 2 introduces class-weighted training to correct the Fake-detection bias present in v1. The trained weights (`artifact_lens_final.h5`) power the FastAPI inference backend in the ArtifactLens web application.

---

## Version Comparison

| Metric | v1 (Baseline) | v2 (Production) |
|---|---|---|
| **Overall Accuracy** | 73.95% | **82.99%** |
| **Real Detection** | ~65% | **87.25%** |
| **Fake Detection** | ~87% | **78.74%** |
| **Bias** | Strong Fake bias | Balanced |
| **Training Change** | Standard loss | Class-weighted loss |
| **Architecture Change** | — | None (same ResNet50) |

> **Key insight:** +9.04% accuracy improvement with zero architecture changes. The entire gain came from class-weighted training — one technique applied to the loss function.

---

## Training Results (v2 — Production)

| Metric | Value |
|---|---|
| **Architecture** | ResNet50 (ImageNet pretrained) |
| **Training Images** | 110,911 |
| **Validation Images** | 27,727 |
| **Total Dataset** | 138,638 face images |
| **Best Val Accuracy** | **82.99%** |
| **Total Epochs** | 20 |
| **Batch Size** | 16 |
| **Hardware** | NVIDIA RTX A6000 (48GB VRAM) |
| **Precision** | Mixed Float16 (fp16 compute, fp32 output) |
| **Class Weights** | Computed via `sklearn.utils.class_weight.compute_class_weight` |

### v2 Accuracy Progression

| Epoch | Train Acc | Val Acc |
|---|---|---|
| 1 | 57.12% | 61.04% |
| 5 | 70.43% | 72.18% |
| 10 | 74.82% | 76.91% |
| 15 | 79.36% | 81.44% |
| **Best** | — | **82.99% ✅** |
| 20 | 80.11% | 82.47% |

---

## Why v1 Was Biased (And How v2 Fixed It)

### The Problem
v1 trained with standard cross-entropy loss treated every wrong prediction with equal penalty. GAN-generated fake images are statistically uniform — they all share the same artifact signature. Real images are chaotic — different cameras, lighting, ethnicities, compression. The model found it far easier to learn one consistent fake pattern than thousands of real variations, resulting in ~87% fake detection but only ~65% real detection.

### The Fix — Class-Weighted Loss
Class weights assign a higher penalty for misclassifying the harder class. Computed using:

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=train_labels
)
# Result: {0 (Fake): 1.008, 1 (Real): 0.992}
```

Applied during training:
```python
model.fit(
    train_generator,
    class_weight={0: 1.008, 1: 0.992},
    ...
)
```

The slightly higher penalty on Real misclassifications forces the model to pay more attention to real image features, breaking the bias without any architecture changes.

---

## Model Architecture

```
Input (224 × 224 × 3)
        │
        ▼
ResNet50 (pretrained on ImageNet)
base_model.trainable = False
        │
        ▼
GlobalAveragePooling2D
        │
        ▼
Dense(512, activation='relu')
        │
        ▼
Dropout(0.5)
        │
        ▼
Dense(1, activation='sigmoid', dtype='float32')
        │
        ▼
Binary Output: Real / Fake
```

**Key design decisions:**
- ResNet50 base frozen — leverages ImageNet features, prevents overfitting on limited data
- `mixed_float16` precision — up to 3× faster training on Ampere architecture GPUs
- `dtype='float32'` on output layer — numerical stability for sigmoid with mixed precision; float16 loses precision for small probability values near 0
- `ModelCheckpoint` with `save_best_only=True` — only saves when val_accuracy improves
- `EarlyStopping` (patience=3) — halts training and restores best weights if val_loss stops improving

---

## Two-Phase Pipeline

### Phase 1 — Preprocessing

- Raw face images collected across real and GAN-generated fake categories
- MTCNN face detection applied to every image
- Eye keypoints extracted → rotation angle computed → affine alignment applied
- Faces cropped with 30px margin and resized to 224×224
- Images smaller than 80×80 upscaled before detection to improve MTCNN reliability
- Successfully processed: **138,638 images** (724 real + 640 fake skipped — no face detected)
- Processed data split into `real/` and `fake/` directories
- 200-image test samples randomly sampled per class for quick evaluation

### Phase 2 — Training

- `ImageDataGenerator` with augmentation (rotation ±20°, horizontal flip, rescale 1/255)
- 80/20 train/validation split
- `Adam` optimizer at `lr=0.0001`
- Class-weighted loss applied (v2 only)
- `EarlyStopping` (patience=3, monitors val_loss)
- `ModelCheckpoint` saves best weights to `artifact_lens_final.h5`

---

## Inference Pipeline

The `inference_copy.py` module handles production inference with a full face preprocessing step before the model sees the image:

```
Raw image bytes received
        │
        ▼
cv2 decode → BGR image
        │
        ▼
MTCNN face detection
(returns bounding box + eye keypoints)
        │
        ▼
Eye alignment
(rotation matrix from left/right eye angle using arctan2)
        │
        ▼
Face crop with 30px margin
+ resize to 224×224
        │
        ▼
Normalize pixel values to [0, 1]
        │
        ▼
ResNet50 model.predict()
        │
        ▼
Threshold @ 0.5
├── > 0.5 → Fake (confidence = raw score)
└── ≤ 0.5 → Real (confidence = 1 - raw score)
```

**Why 0.5 threshold (unlike v1)?**
v1 required a lowered threshold of 0.35 to compensate for the model's Fake-detection bias — most real image scores clustered below 0.5 even for genuine photos. v2's class-weighted training corrects this bias directly during training, so the standard 0.5 threshold works correctly without any post-hoc compensation.

---

## Getting Started

### Prerequisites
- Python 3.9
- CUDA-compatible GPU recommended (CPU inference works but is slow)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Corerishi/ArtifactLens-Model.git
cd ArtifactLens-Model

# 2. Create environment
conda create -n artifact_lens_env python=3.9
conda activate artifact_lens_env

# 3. Install dependencies
pip install -r requirements.txt
```

### Run Inference

```python
from inference_copy import analyze_image

with open("test_image.jpg", "rb") as f:
    image_bytes = f.read()

result = analyze_image(image_bytes)
print(result)
# Output: {"label": "Fake", "confidence": 0.87}
```

### Retrain the Model

1. Place processed face images in:
```
Phase1_Preprocessing/processed_data/
    ├── real/
    └── fake/
```
2. Open and run `Phase2_Training/train_model.ipynb`

---

## Project Structure

```
ArtifactLens-Model/
├── Phase1_Preprocessing/
│   ├── Phase1_Testing.ipynb       # Data preparation + sampling
│   └── failed_images.log          # Images that failed preprocessing
│
├── Phase2_Training/
│   ├── train_model.ipynb          # Full training pipeline (v1 + v2)
│   ├── commnts.ipynb              # Experiment notes
│   ├── artifact_lens_final.h5     # Best v2 model weights (82.99% val accuracy)
│   ├── test_best_model.h5         # v1 baseline checkpoint (73.95% val accuracy)
│   └── test_sample/               # 400-image evaluation set (200 real, 200 fake)
│
├── inference_copy.py              # Production inference module (used by ArtifactLens API)
├── demo.py                        # Standalone demo script
├── requirements.txt
└── README.md
```

---

## Key Dependencies

```
tensorflow==2.x
tf-keras
mtcnn
opencv-python
numpy
scikit-learn    # for compute_class_weight (v2 training)
```

---

## Training Hardware

Model training was conducted on an **NVIDIA RTX A6000 GPU** (48GB VRAM, Compute Capability 8.6) accessed via university high-performance computing infrastructure. Mixed float16 precision training was enabled, leveraging the A6000's Tensor Cores (Compute Capability ≥7.0) to significantly accelerate matrix operations.

| Hardware Spec | Value |
|---|---|
| GPU | NVIDIA RTX A6000 |
| VRAM | 48GB |
| Compute Capability | 8.6 (Ampere) |
| Training Precision | Mixed Float16 |
| RAM | 32GB minimum (64GB recommended) |
| Storage | 100GB+ for dataset + checkpoints |

---

## Related Repository

The trained model from this repo is deployed inside:

**[ArtifactLens](https://github.com/Corerishi/ArtifactLens)** — FastAPI + React + MongoDB Atlas deepfake detection web app

---

## Model Weights

The trained model weights (`artifact_lens_final.h5`) are **not included** in this repository.  
To obtain the weights for research or academic purposes, please reach out via:

[LinkedIn](https://linkedin.com/in/rishi-raj-9110a824a)

---

## Author

**Rishi Raj**  
MCA — CHRIST (Deemed to be University)  
[LinkedIn](https://linkedin.com/in/rishi-raj-9110a824a) · [GitHub](https://github.com/Corerishi)

---

## License

This project is licensed under the [MIT License](LICENSE).
