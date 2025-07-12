import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Potato Disease Classifier", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("saved_models/1.keras")

model = load_model()

# Load the trained model
# model = tf.keras.models.load_model("saved_models/1.keras")
st.text("✅ Model loaded successfully.")


# Define class names
CLASS_NAMES = ["Potato___Early_blight", "Potato___healthy", "Potato_Late___blight"]

# Disease info
DISEASE_INFO = {
    "Potato___Early_blight": "🟠 Early Blight: Caused by *Alternaria solani*. Symptoms include dark spots with concentric rings. Manage with fungicides and crop rotation.",
    "Potato___healthy": "🟢 Healthy: The leaf appears normal without any symptoms of blight or infection. Maintain good practices to keep it that way!",
    "Potato___Late_blight": "🔴 Late Blight: Caused by *Phytophthora infestans*. Symptoms include irregular water-soaked lesions. Requires immediate treatment!"
}

# Function to preprocess image
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")  # Force RGB
    image = image.resize((256, 256))  # Same size as used during training
    img_array = np.array(image) / 255.0  # Normalize like training
    return np.expand_dims(img_array, axis=0), image


# Streamlit UI
# st.set_page_config(page_title="Potato Disease Classifier", layout="centered")

st.title("🥔 Potato Disease Classification")
st.markdown("""
Upload a potato leaf image and detect whether it has a disease like Early or Late Blight.
This tool uses a Convolutional Neural Network (CNN) model built with TensorFlow.
""")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Leaf Image", use_column_width=True)
    st.write("Filename:", uploaded_file.name)

    # Preprocess and predict
    img_batch, display_image = preprocess_image(uploaded_file)
    predictions = model.predict(img_batch)
    st.write("Raw prediction probabilities:", predictions[0])
    st.write("Raw Prediction Output:", predictions)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    st.markdown("---")
    st.subheader("📌 Prediction")
    st.write(f"**Predicted Class:** `{predicted_class}`")
    st.write(f"**Confidence:** `{confidence:.2%}`")

    # Display info card
    st.markdown("### 📝 Details")
    st.info(DISEASE_INFO[predicted_class])
