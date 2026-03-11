# 🎤 Speech Emotion Recognition (SER)

A Deep Learning-based Speech Emotion Recognition system that classifies human speech into **8 emotion categories** using advanced audio feature extraction and a 1D CNN model.

---

## 🚀 Overview

This project combines multiple public speech datasets and applies feature engineering + deep learning to detect emotions from voice samples.

The system supports:
- 🎧 WAV file upload  
- 🎙️ Real-time microphone recording  
- 📊 Emotion prediction with confidence scores  

---

## 🎭 Emotions Detected

- Angry  
- Calm  
- Disgust  
- Fear  
- Happy  
- Neutral  
- Sad  
- Surprise  

---

## 🧠 Model Architecture

- 1D Convolutional Neural Network (CNN)
- Batch Normalization
- MaxPooling
- Dropout Regularization
- Class Weight Balancing
- Early Stopping + Learning Rate Scheduler

**Final Test Accuracy: ~67% (8-class classification)**

---

## 📂 Datasets Used

Merged and unified 12,000+ audio samples from:

- RAVDESS  
- CREMA-D  
- TESS  
- SAVEE  

After augmentation, dataset size increased to **36,000+ samples**.

---

## 🎛️ Feature Engineering

Extracted using Librosa:

- MFCC (40 coefficients)  
- Chroma STFT  
- Mel Spectrogram  
- Zero Crossing Rate (ZCR)  
- RMS Energy  

---

## 🔄 Data Augmentation

To improve generalization:

- Noise Injection  
- Time Stretching  
- Pitch Shifting  

---

## 🛠 Tech Stack

- Python  
- TensorFlow / Keras  
- Librosa  
- Scikit-learn  
- NumPy  
- Seaborn  
- Streamlit  

---

## 📦 Installation

```bash
git clone https://github.com/Rishabh-Shukla-15/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 📁 Saved Model Files

- `ser_model.keras`
- `scaler.pkl`
- `encoder.pkl`

---

## 👨‍💻 Author

Rishabh Shukla  
B.Tech – Artificial Intelligence & Machine Learning  
NIT Kurukshetra  
# Speech-Emotion-Recognition
