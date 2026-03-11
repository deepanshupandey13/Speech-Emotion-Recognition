import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import joblib
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os

# ------------------ LOAD MODEL & OBJECTS ------------------

model = tf.keras.models.load_model("ser_model.keras")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

# ---- Hardcoded values (same as training) ----
DURATION = 2.5
OFFSET = 0.6
N_MFCC = 40
SAMPLE_RATE = 22050

# ------------------ FEATURE EXTRACTION ------------------

def extract_features(data, sr):
    result = np.array([])

    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    stft = np.abs(librosa.stft(y=data))

    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    result = np.hstack((result, chroma))

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=N_MFCC).T, axis=0)
    result = np.hstack((result, mfcc))

    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128).T, axis=0)
    result = np.hstack((result, mel))

    return result

# ------------------ PREDICTION FUNCTION ------------------

def predict_emotion(audio_path):
    data, sr = librosa.load(audio_path, duration=DURATION, offset=OFFSET)

    features = extract_features(data, sr)
    features = scaler.transform([features])
    features = np.expand_dims(features, axis=2)

    prediction = model.predict(features)
    emotion = encoder.inverse_transform(prediction)

    return emotion[0][0], prediction[0]

# ------------------ STREAMLIT UI ------------------

st.set_page_config(page_title="Speech Emotion Recognition")

st.title("🎤 Speech Emotion Recognition")
st.write("Upload a WAV file or record your voice.")

# ------------------ FILE UPLOAD ------------------

uploaded_file = st.file_uploader("Upload Audio File (.wav only)", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(uploaded_file)

    if st.button("Predict Emotion"):
        emotion, probs = predict_emotion(tmp_path)

        st.success(f"Predicted Emotion: {emotion}")

        st.subheader("Confidence Scores:")
        emotions = encoder.categories_[0]
        for e, p in zip(emotions, probs):
            st.write(f"{e} : {round(float(p)*100, 2)} %")

# ------------------ RECORD FROM MICROPHONE ------------------

st.subheader("Or Record Your Voice")

if st.button("Start Recording (3 sec)"):
    st.info("Recording... Speak now!")

    recording = sd.rec(int(3 * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE,
                       channels=1)
    sd.wait()

    temp_audio = "recorded.wav"
    write(temp_audio, SAMPLE_RATE, recording)

    st.audio(temp_audio)

    emotion, probs = predict_emotion(temp_audio)

    st.success(f"Predicted Emotion: {emotion}")

    st.subheader("Confidence Scores:")
    emotions = encoder.categories_[0]
    for e, p in zip(emotions, probs):
        st.write(f"{e} : {round(float(p)*100, 2)} %")