# Audio-Emotion-Recognition
# Speech Emotion Recognition using Wav2Vec2

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/Dpngtm)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VNhIjY7gW29d0uKGNDGN0eOp-pxr_pFL)  

## 📌 Project Overview
This project focuses on **Speech Emotion Recognition (SER)**, leveraging the power of Self-Supervised Learning (SSL). The goal is to classify raw audio signals into distinct emotional categories (e.g., Happy, Sad, Angry, Neutral) by fine-tuning the **Wav2Vec2** architecture.

The model was trained on a composite dataset to ensure robustness across different voice types and recording conditions, achieving a test accuracy of **~79.6%**.

🔗 **Live Demo:** [Try the Gradio App on Hugging Face Spaces](https://huggingface.co/spaces/Dpngtm/Audio-Emotion-Recognition)

## 📊 Dataset Composition
To create a diverse and generalized model, I combined four prominent audio datasets:
* **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
* **TESS** (Toronto Emotional Speech Set)
* **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset)
* **SAVEE** (Surrey Audio-Visual Expressed Emotion)

*Preprocessing involved audio normalization, resampling to 16kHz, and categorical label alignment.*

## 🛠️ Tech Stack & Methodology
* **Model Architecture:** `facebook/wav2vec2-base` (Fine-tuned for Classification Head).
* **Frameworks:** PyTorch, Hugging Face Transformers.
* **Audio Processing:** Librosa.
* **Deployment:** Gradio (Web Interface).

### Key Technical Challenge
Standard speech models are optimized for ASR (transcription). The challenge was to freeze the feature extractor of Wav2Vec2 and retrain the classification head to detect *prosodic features* (tone, pitch, intensity) rather than linguistic content.

## 📈 Results
The model was evaluated on a held-out test set derived from the combined dataset.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **79.57%** |
| **F1 Score** | **79.43%** |

## 🚀 How to Run

### Option 1: Google Colab (Recommended)
You can run the full training and inference pipeline directly in your browser without local setup:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VNhIjY7gW29d0uKGNDGN0eOp-pxr_pFL)

### Option 2: Local Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/DGautam11/Audio-Emotion-Recognition.git](https://github.com/DGautam11/Audio-Emotion-Recognition.git)
   cd Audio-Emotion-Recognition
