import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from pydub import AudioSegment
import os
import gdown
import cv2

# Constants
MODEL_URL = "https://drive.google.com/uc?export=download&id=1DK8yTXMdgSmjWAYKxCZP1F7HrDITMxQj"
MODEL_PATH = 'combinemodel.h5'
CLASS_LABELS = ['Normal', 'Inner', 'Roller', 'Outer']
EXPECTED_SHAPE = (1080, 720, 3)  # Model's expected shape for mel spectrogram input
FEATURE_SHAPE = 5  # Model's expected shape for features input

# Download the model if not present
if not os.path.exists(MODEL_PATH):
    st.write("Downloading the model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.write("Model downloaded successfully!")

# Load the model
model = load_model(MODEL_PATH)

def convert_to_wav(mp3_file):
    """
    Convert an MP3 file to WAV format using pydub.
    """
    audio = AudioSegment.from_file(mp3_file, format="mp3")
    wav_file = "temp.wav"
    audio.export(wav_file, format="wav")
    return wav_file

def preprocess_audio(wav_path):
    """
    Preprocess audio file by extracting Mel Spectrogram and resizing it to the model's expected input shape.
    """
    # Placeholder for Mel Spectrogram (replace with real feature extraction logic)
    mel_input = np.random.rand(128, 128, 3)  # Example dummy input

    # Resize to model's expected shape
    mel_input_resized = cv2.resize(mel_input, (EXPECTED_SHAPE[1], EXPECTED_SHAPE[0]))
    return mel_input_resized

def extract_features(wav_path):
    """
    Extract features for the second input of the model.
    Ensure the output shape matches (None, 5).
    """
    # Example: Generate 5 dummy features (Replace with real logic)
    features_input = np.random.rand(FEATURE_SHAPE)  # Replace with real feature extraction logic
    return features_input

# Streamlit App
st.title("Audio Classifier")
st.write("Upload an audio file (MP3 format) to predict its class.")

# File uploader
uploaded_file = st.file_uploader("Upload an Audio File", type=["mp3"])

if uploaded_file is not None:
    # Display the MIME type of the uploaded file
    st.write(f"Detected MIME type: {uploaded_file.type}")

    # Save the uploaded file temporarily
    tmp_file_path = "temp_audio.mp3"
    with open(tmp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Convert MP3 to WAV
    st.write("Converting MP3 to WAV...")
    wav_path = convert_to_wav(tmp_file_path)
    st.audio(wav_path, format='audio/wav')
    
    # Process the audio
    st.write("Processing the audio file...")
    mel_input = preprocess_audio(wav_path)
    features_input = extract_features(wav_path)

    # Add batch dimensions
    mel_input = np.expand_dims(mel_input, axis=0)  # Shape: (1, 1080, 720, 3)
    features_input = np.expand_dims(features_input, axis=0)  # Shape: (1, 5)

    # Debug shapes
    st.write("Shape of mel_input:", mel_input.shape)
    st.write("Shape of features_input:", features_input.shape)

    # Make predictions
    try:
        predictions = model.predict([mel_input, features_input])
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = CLASS_LABELS[predicted_class]
        confidence = np.max(predictions)

        # Display results
        st.success(f"Predicted Class: {predicted_label}")
        st.write(f"Prediction Confidence: {confidence:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
