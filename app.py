import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from scipy.io.wavfile import read
from twilio.rest import Client
import os

# Load the pre-trained model
MODEL_PATH = 'CWRU_CNN_raw_time_domain_data.h5'  # Replace with your actual model path
model = load_model(MODEL_PATH)

# Labels for classification
LABELS = ['Ball_007', 'Ball_014', 'Ball_021', 'IR_007', 'IR_014', 'IR_021', 'Normal', 'OR_007', 'OR_014', 'OR_021']

# Suggestions for each defect type
SUGGESTIONS = {
    'Ball_007': [
        "Condition Monitoring: Use vibration analysis or acoustic emission to detect surface defects.",
        "Proper Lubrication: Apply suitable grease or oil and maintain lubrication schedules.",
        "Clean Environment: Prevent contamination by using seals and keeping the area clean.",
        "Material Upgrade: Use hardened or coated rollers for demanding applications."
    ],
    'Ball_014': [
        "Condition Monitoring: Use vibration analysis or acoustic emission to detect surface defects.",
        "Proper Lubrication: Apply suitable grease or oil and maintain lubrication schedules.",
        "Clean Environment: Prevent contamination by using seals and keeping the area clean.",
        "Material Upgrade: Use hardened or coated rollers for demanding applications."
    ],
    'Ball_021': [
        "Condition Monitoring: Use vibration analysis or acoustic emission to detect surface defects.",
        "Proper Lubrication: Apply suitable grease or oil and maintain lubrication schedules.",
        "Clean Environment: Prevent contamination by using seals and keeping the area clean.",
        "Material Upgrade: Use hardened or coated rollers for demanding applications."
    ],
    'IR_007': [
        "Load Analysis: Ensure proper load distribution and avoid overloading.",
        "Precision Installation: Use tools like hydraulic presses for accurate assembly.",
        "Periodic Inspection: Conduct ultrasonic or magnetic particle testing.",
        "Enhanced Design: Opt for high-quality materials and better manufacturing tolerances."
    ],
    'IR_014': [
        "Load Analysis: Ensure proper load distribution and avoid overloading.",
        "Precision Installation: Use tools like hydraulic presses for accurate assembly.",
        "Periodic Inspection: Conduct ultrasonic or magnetic particle testing.",
        "Enhanced Design: Opt for high-quality materials and better manufacturing tolerances."
    ],
    'IR_021': [
        "Load Analysis: Ensure proper load distribution and avoid overloading.",
        "Precision Installation: Use tools like hydraulic presses for accurate assembly.",
        "Periodic Inspection: Conduct ultrasonic or magnetic particle testing.",
        "Enhanced Design: Opt for high-quality materials and better manufacturing tolerances."
    ],
    'OR_007': [
        "Temperature Control: Use temperature monitoring systems to prevent overheating.",
        "Corrosion Prevention: Apply anti-corrosion coatings and ensure proper sealing.",
        "Dynamic Balancing: Maintain system balance to reduce uneven stresses on the outer race.",
        "Lubricant Selection: Use lubricants suited for the operating environment."
    ],
    'OR_014': [
        "Temperature Control: Use temperature monitoring systems to prevent overheating.",
        "Corrosion Prevention: Apply anti-corrosion coatings and ensure proper sealing.",
        "Dynamic Balancing: Maintain system balance to reduce uneven stresses on the outer race.",
        "Lubricant Selection: Use lubricants suited for the operating environment."
    ],
    'OR_021': [
        "Temperature Control: Use temperature monitoring systems to prevent overheating.",
        "Corrosion Prevention: Apply anti-corrosion coatings and ensure proper sealing.",
        "Dynamic Balancing: Maintain system balance to reduce uneven stresses on the outer race.",
        "Lubricant Selection: Use lubricants suited for the operating environment."
    ]
}


# GIF Mapping
GIFS = {
    'Ball_007': 'outer.gif',
    'Ball_014': 'outer.gif',
    'Ball_021': 'outer.gif',
    'IR_007': 'inner.gif',
    'IR_014': 'inner.gif',
    'IR_021': 'inner.gif',
    'OR_007': 'outer.gif',
    'OR_014': 'outer.gif',
    'OR_021': 'outer.gif'
}

# Twilio SMS function
def send_sms(prediction):
    """
    Sends an SMS with the prediction results to a predefined phone number.
    """
    # Twilio credentials from Streamlit Secrets
    account_sid = st.secrets["twilio"]["account_sid"]
    auth_token = st.secrets["twilio"]["auth_token"]
    twilio_phone_number = st.secrets["twilio"]["twilio_phone_number"]
    permanent_phone_number = st.secrets["twilio"]["permanent_phone_number"]

    # Create a Twilio client
    client = Client(account_sid, auth_token)

    # Send the message
    message = client.messages.create(
        body=f"Bearing Fault Prediction: {prediction}",
        from_=twilio_phone_number,
        to=permanent_phone_number
    )

    print(f"Message sent! SID: {message.sid}")

# Function to preprocess the audio file
def preprocess_audio(file_path):
    sample_rate, audio_data = read(file_path)

    # Convert to mono if stereo
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Normalize the audio data
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Ensure the length is 32x32 = 1024
    total_elements = 32 * 32
    if len(audio_data) < total_elements:
        padded_data = np.pad(audio_data, (0, total_elements - len(audio_data)), mode='constant')
    elif len(audio_data) > total_elements:
        padded_data = audio_data[:total_elements]
    else:
        padded_data = audio_data

    # Reshape into 32x32 array
    reshaped_data = padded_data.reshape(32, 32)
    
    # Prepare for the model
    input_data = np.expand_dims(reshaped_data, axis=0)  # Add batch dimension
    input_data = np.expand_dims(input_data, axis=-1)    # Add channel dimension for CNN

    return input_data

# Streamlit UI
st.title("BEARING DEFECT CLASSIFIER")
st.markdown("<br><br>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload an MP3 or WAV file", type=["mp3", "wav"])

if uploaded_file is not None:
    temp_file = "temp_audio_file.wav"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.read())

    # Play uploaded audio file
    st.audio(uploaded_file)

    try:
        input_data = preprocess_audio(temp_file)
        predictions = model.predict(input_data)
        predicted_class_index = np.argmax(predictions, axis=-1)
        predicted_label = LABELS[predicted_class_index[0]]
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.success(f"Predicted Defect: {predicted_label}")

        # Display GIF if available for the prediction
        if predicted_label in GIFS:
            gif_path = GIFS[predicted_label]
            st.image(gif_path, caption=f"Visualization for {predicted_label}", use_container_width=True)

        # Show suggestions based on the prediction
        if predicted_label in SUGGESTIONS:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.subheader("Suggestions for Maintenance and Prevention:")
            for suggestion in SUGGESTIONS[predicted_label]:
                st.write(f"🔧 {suggestion}")

        # Send SMS to permanent phone number
        send_sms(predicted_label)
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info("Prediction sent via SMS to the predefined number!")
    except Exception as e:
        st.error(f"Error processing the file: {e}")

    # Clean up the temporary file
    os.remove(temp_file)
