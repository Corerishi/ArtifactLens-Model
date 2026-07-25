#  ArtifactLens — Model Training & Inference

> **The ML backbone of ArtifactLens** — ResNet50-based deepfake detection model trained on 138K+ face images across 20 epochs on an NVIDIA RTX A6000, achieving **73.95% validation accuracy**.

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![ResNet50](https://img.shields.io/badge/Architecture-ResNet50-blueviolet?style=flat)
![MTCNN](https://img.shields.io/badge/Face%20Detection-MTCNN-0097A7?style=flat)
![GPU](https://img.shields.io/badge/Hardware-NVIDIA%20RTX%20A6000-76B900?style=flat&logo=nvidia&logoColor=white)

> 🔗 This repository contains the training pipeline and inference module for the [ArtifactLens](https://github.com/Corerishi/ArtifactLens) full-stack deepfake detection web app.

---

## What is this?

This repo documents the complete **model development pipeline** for ArtifactLens — from dataset preparation and preprocessing (Phase 1) to model training, checkpointing, and inference (Phase 2).

The model is a **fine-tuned ResNet50** classifier trained to distinguish real vs. AI-generated (deepfake) face images. The trained weights (`artifact_lens_final.h5`) power the FastAPI inference backend in the ArtifactLens web application.

---

## Training Results

| Metric | Value |
|---|---|
| **Architecture** | ResNet50 (ImageNet pretrained) |
| **Training Images** | 110,911 |
| **Validation Images** | 27,727 |
| **Total Dataset** | 138,638 face images |
| **Best Val Accuracy** | **73.95%** (Epoch 18) |
| **Best Val Loss** | 0.5319 |
| **Total Epochs** | 20 |
| **Batch Size** | 16 |
| **Hardware** | NVIDIA RTX A6000 (48GB VRAM) |
| **Precision** | Mixed Float16 (fp16 compute, fp32 output) |

### Accuracy Progression

| Epoch | Train Acc | Val Acc |
|---|---|---|
| 1 | 55.87% | 59.71% |
| 5 | 68.92% | 69.74% |
| 10 | 71.31% | 71.62% |
| 15 | 72.30% | 73.14% |
| **18** | **72.80%** | **73.95% ✅ Best** |
| 20 | 72.88% | 73.34% |

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
- `dtype='float32'` on output layer — numerical stability for sigmoid with mixed precision
- `ModelCheckpoint` with `save_best_only=True` — only saves when val_accuracy improves

---

## Two-Phase Pipeline

### Phase 1 — Preprocessing
- Raw video frames extracted and processed
- Face regions detected and cropped
- Processed data split into `real/` and `fake/` directories
- 200-image test samples randomly sampled per class for quick evaluation

### Phase 2 — Training
- `ImageDataGenerator` with augmentation (rotation ±20°, horizontal flip, rescale)
- 80/20 train/validation split
- `Adam` optimizer at `lr=0.0001`
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
(rotation matrix from left/right eye angle)
        │
        ▼
Face crop with 30px margin
+ resize to 224×224
        │
        ▼
Normalize to [0, 1]
        │
        ▼
ResNet50 model.predict()
        │
        ▼
Threshold @ 0.35
├── > 0.35 → Real (confidence = raw score)
└── ≤ 0.35 → Fake (confidence = 1 - raw score)
```

**Why threshold = 0.35 instead of 0.5?**  
The model was observed to be conservative in predicting "Real" — most scores clustered below 0.5 even for genuine images. Lowering the threshold to 0.35 corrects for this bias and produces more balanced Real/Fake predictions in production.

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
│   ├── train_model.ipynb          # Full training pipeline
│   ├── commnts.ipynb              # Experiment notes
│   ├── artifact_lens_final.h5     # Best model weights (saved by checkpoint)
│   ├── test_best_model.h5         # Alternate checkpoint
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
```

---

## Related Repository

The trained model from this repo is deployed inside:

 **[ArtifactLens](https://github.com/Corerishi/ArtifactLens)** — FastAPI + React + MongoDB Atlas deepfake detection web app

---

## Model Weights

The trained model weights (`artifact_lens_final.h5`) are **not included** in this repository.  
To obtain the weights for research or academic purposes, please reach out via:

 [LinkedIn](https://linkedin.com/in/rishi-raj-9110a824a)

## Author

**Rishi Raj**  
MCA — CHRIST (Deemed to be University)  
[LinkedIn](https://linkedin.com/in/rishi-raj-9110a824a) · [GitHub](https://github.com/Corerishi)

---

## License

This project is licensed under the [MIT License](LICENSE).
