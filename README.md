# Speech Emotion Recognition 🎤

A deep learning system that analyzes **human speech signals** and predicts the **underlying emotion** present in the audio.

The model learns emotional patterns from voice features and classifies speech into **8 emotion categories**.

---

## Project Idea

Human speech carries emotional information beyond just words.  
This project aims to automatically **detect emotions from voice recordings** using signal processing and deep learning.

The system can analyze an input audio clip and determine the speaker's emotional state.

---

## Emotions Recognized

The model predicts the following emotions:

- Angry
- Calm
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

---

## System Workflow

1. Audio input is provided through file upload or microphone recording.
2. Important audio features are extracted using signal processing techniques.
3. The extracted features are passed to a trained **1D CNN model**.
4. The model outputs the predicted emotion along with confidence scores.

---

## Dataset

The training data is created by combining multiple public emotion speech datasets:

- RAVDESS
- CREMA-D
- TESS
- SAVEE

Total samples before augmentation: **12,000+**

After applying augmentation techniques, the dataset increased to **36,000+ samples**.

---

## Feature Extraction

Audio features are extracted using the **Librosa** library.

Features used:

- MFCC (Mel Frequency Cepstral Coefficients)
- Chroma Features
- Mel Spectrogram
- Zero Crossing Rate
- RMS Energy

These features capture important characteristics of speech such as **tone, pitch, energy, and frequency distribution**.

---

## Data Augmentation

To improve generalization and reduce overfitting, the dataset is augmented using:

- Noise Injection
- Time Stretching
- Pitch Shifting

---

## Model Architecture

The emotion classifier is built using a **1D Convolutional Neural Network**.

Main components:

- Convolution layers for feature learning
- Batch normalization
- Max pooling layers
- Dropout for regularization

Training improvements include:

- Class weight balancing
- Early stopping
- Learning rate scheduling

Final test accuracy: **~67% for 8-class classification**

---

## Tech Stack

- Python
- TensorFlow / Keras
- Librosa
- Scikit-learn
- NumPy
- Streamlit

---


## Running the Project

Clone the repository:

```bash

### install dependency
pip install -r requirements.txt
streamlit run app.py



git clone https://github.com/yourusername/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition
