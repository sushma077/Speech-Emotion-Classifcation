import streamlit as st
import numpy as np
import librosa
import torch
import torch.nn as nn
import pickle


@st.cache_resource
def load_model():
    with open("best_emotion_model.pth", "rb") as f:
        model = pickle.load(f)
    model.eval()
    return model

model = load_model()

import librosa
import numpy as np

def extract_features(path):
    y, sr = librosa.load(path, duration=3, offset=0.5)

    # MFCC
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

    # Chroma
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)

    # Mel Spectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)

    # Zero-Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)

    # RMS Energy
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)

    # Spectral Contrast
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)

    # Tonnetz
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)

    # Combine features
    features = np.hstack([mfccs, chroma, mel, zcr, rms, contrast, tonnetz])

    # Pad or trim to 180
    if len(features) < 180:
        features = np.pad(features, (0, 180 - len(features)))
    else:
        features = features[:180]

    return features

# STREAMLIT UI 

st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")

st.title("🎙️ Speech Emotion Recognition")
st.markdown("Upload a `.wav` audio file, and the model will predict the **emotion** expressed in the speech.")

uploaded_file = st.file_uploader("Upload your audio file (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    try:
        input_tensor = extract_features("temp.wav")
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)  # shape: [1, 6]
            prediction = torch.argmax(output, dim=1).item()

        label_map = {
            0: "calm",
            1: "happy",
            2: "sad",
            3: "angry",
            4: "fearful",
            5: "disgust",
            
        }

        emotion = label_map.get(prediction, f"Class {prediction}")
        st.success(f"🧠 **Predicted Emotion**: `{emotion.upper()}`")

    except Exception as e:
        st.error(f"❌ Prediction Failed: {e}")
else:
    st.info("Please upload a `.wav` file to begin.")
