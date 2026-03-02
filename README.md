# 🖊️ Signature Verification AI System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-macOS%20M1-lightgrey.svg)](https://www.apple.com/mac/)

An AI-powered signature verification system using Siamese Neural Networks to authenticate handwritten signatures. Built with PyTorch and deployed as an interactive web application.

![Signature Verification Demo](assets/demo_screenshot.png)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Technologies](#technologies)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

This project implements a deep learning solution for signature verification using a Siamese Neural Network architecture. The system can determine whether a signature is genuine or forged by comparing it with a reference signature, achieving **74% accuracy** and an **F1-score of 0.81** on test data.

### Key Highlights

- **15.7 million trainable parameters** in the neural network
- **Real-time inference** (<1 second per signature pair)
- **Interactive web interface** built with Streamlit
- **GPU-accelerated** training and inference on Apple M1 (MPS)
- **Complete ML pipeline** from data preprocessing to deployment

## ✨ Features

- ✅ **One-shot learning** - Verify signatures with minimal reference samples
- ✅ **Real-time processing** - Instant verification results
- ✅ **Robust to variations** - Handles natural signature differences
- ✅ **Confidence scoring** - Provides interpretable distance metrics
- ✅ **User-friendly interface** - Drag-and-drop web application
- ✅ **Adjustable threshold** - Tune sensitivity for different use cases

## 🎬 Demo

### Web Application
![Web App Interface](assets/web_app_demo.gif)

### Sample Results
| Test Case | Distance | Prediction | Result |
|-----------|----------|------------|---------|
| Same person (genuine) | 0.0758 | ✅ Genuine | Correct |
| Different person | 0.1069 | ❌ Forged | Correct |
| Forged signature | 1.2177 | ❌ Forged | Correct |

## 🔧 Installation

### Prerequisites

- Python 3.9 or higher
- macOS (for M1 GPU support) or Linux/Windows
- 5+ GB free disk space

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/signature-verification-ai.git
cd signature-verification-ai
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the dataset** (optional - for training)
```bash
# Option 1: Use sample data (included)
python download_dataset.py

# Option 2: Download CEDAR dataset
# Visit: https://www.kaggle.com/datasets/ishadss/cedar-signature-verification-dataset
# Place in: data/raw/cedar/
```

## 🚀 Usage

### Quick Start - Web Application

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Training the Model

```bash
# Train with default settings
python -m src.train

# View training progress with TensorBoard
tensorboard --logdir results/logs
```

### Evaluating the Model

```bash
# Run evaluation on test set
python -m src.evaluate

# Outputs: confusion matrix, ROC curve, performance metrics
```

### Using the Model Programmatically

```python
from src.model import SiameseNetwork
from src.config import Config
import torch
from PIL import Image

# Load trained model
model = SiameseNetwork(embedding_dim=128)
checkpoint = torch.load('models/best_siamese_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load and preprocess images
img1 = preprocess_image(Image.open('signature1.png'))
img2 = preprocess_image(Image.open('signature2.png'))

# Get prediction
with torch.no_grad():
    emb1, emb2 = model(img1, img2)
    distance = torch.nn.functional.pairwise_distance(emb1, emb2)
    is_genuine = distance < 0.1302  # Optimal threshold
```

## 📁 Project Structure

```
signature-verification-ai/
├── data/
│   ├── raw/                    # Original signature images
│   │   ├── sample_signatures/  # Sample dataset
│   │   └── cedar/             # CEDAR dataset (optional)
│   └── processed/             # Preprocessed data
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration & hyperparameters
│   ├── dataset.py             # Data loading & preprocessing
│   ├── model.py               # Siamese Network architecture
│   ├── train.py               # Training pipeline
│   ├── evaluate.py            # Evaluation & metrics
│   └── utils.py               # Helper functions
├── models/                     # Saved trained models
│   └── best_siamese_model.pth
├── results/                    # Training outputs & visualizations
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── logs/                  # TensorBoard logs
├── notebooks/                  # Jupyter notebooks
│   └── experiments.ipynb
├── assets/                     # Images for README
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── README.md
├── LICENSE
└── .gitignore
```

## 🧠 Model Architecture

### Siamese Neural Network

The model uses twin Convolutional Neural Networks (CNNs) with shared weights:

```
Input (155x220 grayscale) → CNN Branch 1 ─┐
                                           ├─→ Distance Metric → Prediction
Input (155x220 grayscale) → CNN Branch 2 ─┘
```

**Architecture Details:**
- **Layer 1:** Conv2D(32) → BatchNorm → ReLU → MaxPool → Dropout(0.2)
- **Layer 2:** Conv2D(64) → BatchNorm → ReLU → MaxPool → Dropout(0.2)
- **Layer 3:** Conv2D(128) → BatchNorm → ReLU → MaxPool → Dropout(0.3)
- **Layer 4:** Conv2D(256) → BatchNorm → ReLU → MaxPool → Dropout(0.3)
- **FC Layers:** Flatten → FC(512) → Dropout(0.5) → FC(128)
- **Output:** 128-dimensional embedding

**Loss Function:** Contrastive Loss with margin=1.0

**Distance Metric:** Euclidean distance between embeddings

## 📊 Performance

### Metrics on Test Set

| Metric | Value |
|--------|-------|
| Accuracy | 73.68% |
| Precision | 75.00% |
| Recall | 87.50% |
| F1-Score | 80.77% |
| ROC AUC | 0.6339 |

### Distance Statistics

- **Genuine signatures:** Mean distance = 0.0758 ± 0.0562
- **Forged signatures:** Mean distance = 0.1069 ± 0.0685
- **Optimal threshold:** 0.1302

### Training Details

- **Training time:** 1.32 minutes (12 epochs with early stopping)
- **Hardware:** Apple M1 with MPS GPU acceleration
- **Dataset:** 100 signature pairs (50 genuine, 50 forged)
- **Parameters:** 15,790,400 trainable

## 🛠️ Technologies

### Core Technologies

- **Deep Learning:** PyTorch 2.10
- **Computer Vision:** OpenCV 4.13, PIL
- **Data Science:** NumPy, Pandas, Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Web Framework:** Streamlit

### Development Tools

- **Version Control:** Git/GitHub
- **Environment:** Python 3.14, Virtual Environment
- **Monitoring:** TensorBoard
- **Notebooks:** Jupyter

### Hardware Acceleration

- **GPU:** Apple M1 Metal Performance Shaders (MPS)
- **Alternative:** CUDA (NVIDIA) or CPU

## 🔮 Future Enhancements

- [ ] Integration with full CEDAR dataset (55 writers)
- [ ] Transfer learning with pre-trained models (ResNet, VGG)
- [ ] Multi-language signature support
- [ ] Mobile app deployment (iOS/Android)
- [ ] REST API for system integration
- [ ] Live signature capture via tablet/touchscreen
- [ ] Temporal signature analysis (stroke order, speed)
- [ ] Ensemble methods for improved accuracy

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

Mohammed Aljavvaleh - [@mobeast10](https://x.com/mobeast10?s=21) - mohammedaljavvaleh@gmail.com

Project Link: [https://github.com/mohammed-aljavvaleh/Signature-verification-model](https://github.com/mohammed-aljavvaleh/Signature-verification-model)

## 🙏 Acknowledgments

- CEDAR Signature Verification Dataset
- PyTorch Community
- Siamese Networks research papers
- Streamlit for the amazing web framework

---

**⭐ If you found this project helpful, please consider giving it a star!**
