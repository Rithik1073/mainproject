import streamlit as st
import numpy as np
import os
import gdown
from tensorflow.keras.models import load_model
from pydub import AudioSegment
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io import savemat
import tempfile

# Path to save model
MODEL_PATH = "combinemodel.h5"

# Check if the model file already exists locally, if not, download it
if not os.path.exists(MODEL_PATH):
    # Google Drive shareable link
    file_id = '1DK8yTXMdgSmjWAYKxCZP1F7HrDITMxQj'
    url = f'https://drive.google.com/uc?id={file_id}'

    
    gdown.download(url, MODEL_PATH, quiet=False)

# Load the model
model = load_model(MODEL_PATH)

# Class labels
class_labels = {0: "Infected", 1: "Healthy"}

# Streamlit UI
st.title("Ball bearing damage detector 🎶")
st.write("Upload an audio file (MP3).")

# Image upload widget
uploaded_file = st.file_uploader("Upload an Audio File", type=["mp3"])

if uploaded_file is not None:
    # Display the uploaded file's name
    st.write("Processing...")

    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Convert MP3 to WAV using pydub
    audio = AudioSegment.from_mp3(tmp_file_path)
    wav_path = tmp_file_path.replace(".mp3", ".wav")
    audio.export(wav_path, format="wav")
    
    # Load the WAV file using librosa
    y, sr = librosa.load(wav_path, sr=None)  # sr=None to preserve original sampling rate

    # Extract features (e.g., MFCCs or spectrogram)
    # For this example, we'll use MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # 13 MFCCs
    mfccs = np.mean(mfccs, axis=1)  # Take the mean across time frames

    # Reshape the MFCCs to match the input shape expected by the model
    # Assuming the model expects a 2D input: (1, n_features)
    input_features = np.expand_dims(mfccs, axis=0)

    # Optionally, save the MFCCs as a .mat file for later use (if needed)
    mat_filename = "audio_features.mat"
    savemat(mat_filename, {"mfccs": mfccs})

    # Make prediction
    prediction = model.predict(input_features)
    predicted_class = 1 if prediction > 0.5 else 0  # Threshold: 0.5
    result = class_labels[predicted_class]

    # Display the result
    st.success(f"The audio suggests the leaf is **{result}**.")

    # Optionally, display the MFCCs (or spectrogram) as a plot
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title("MFCCs of the Uploaded Audio")
    st.pyplot(plt)
