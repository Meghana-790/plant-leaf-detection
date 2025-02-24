import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
MODEL_PATH = "trained_plant_disease_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names (update according to your dataset)
class_names = ["Healthy", "Early Blight", "Late Blight"]

def preprocess_image(image):
    """Preprocess the uploaded image for model prediction."""
    image = image.convert("RGB")  # Ensure the image is in RGB format
    image = image.resize((128, 128))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Expand dims to match batch size
    return image

# Streamlit UI with Modern Design
st.set_page_config(page_title="Potato Leaf Disease Detector", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        h1 {
            color: #2E8B57;
            text-align: center;
        }
        .stButton>button {
            background-color: #2E8B57;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
        }
        .stFileUploader {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🍃 Potato Leaf Disease Detection 🍃")
st.markdown("Upload an image of a potato leaf to detect if it's healthy or diseased.")

# Upload section
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"], label_visibility="visible")

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file)
    
    with col1:
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)
    
    with col2:
        st.write("🔄 Processing image...")
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)
        
        # Display Results
        st.success(f"✅ Prediction: {predicted_class}")
        st.write(f"🎯 Confidence: {confidence:.2f}")
        
        # Show class probabilities
        st.write("📊 Class Probabilities:")
        for i, class_name in enumerate(class_names):
            st.progress(float(prediction[0][i]))
            st.write(f"{class_name}: {prediction[0][i]:.2%}")

st.markdown("---")
