import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load the trained model with error handling
@st.cache_resource
def load_model():
    try:
        # Verify file exists before loading
        model_path = 'trained_plant_disease_model.keras'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load model once at startup
model = load_model()

# Define class names - should match your trained model's classes
# Example: replace with your actual plant disease classes
CLASS_NAMES = ['Healthy', 'Late blight', 'Early Blight']  # Update these

def preprocess_image(image):
    try:
        # Convert image to RGB if it's in RGBA or other mode
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize the image to match the model's expected input size
        image = image.resize((128, 128))  

        # Convert image to NumPy array and normalize pixel values to [0,1]
        image_array = np.array(image) / 255.0  

        # Ensure the array has the correct shape (batch size, height, width, channels)
        image_array = np.expand_dims(image_array, axis=0)  

        return image_array
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None


# Streamlit UI
st.title("Plant Leaf Disease Detection")
st.write("Upload an image of a plant leaf to detect potential diseases.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        # Prediction button
        if st.button("Predict"):
            if model is None:
                st.error("Model is not loaded. Please check the model file.")
                st.stop()
                
            # Process and predict
            processed_image = preprocess_image(image)
            if processed_image is not None:
                predictions = model.predict(processed_image)
                predicted_class_idx = np.argmax(predictions[0])
                predicted_class = CLASS_NAMES[predicted_class_idx]
                confidence = np.max(predictions[0]) * 100
                
                # Display results
                st.write("### Results")
                st.write(f"*Prediction:* {predicted_class}")
                st.write(f"*Confidence:* {confidence:.2f}%")
                
                # Optional: Show prediction probabilities for all classes
                st.write("#### Prediction Probabilities:")
                for class_name, prob in zip(CLASS_NAMES, predictions[0]):
                    st.write(f"{class_name}: {prob*100:.2f}%")
                    
    except Exception as e:
        st.error(f"Error processing uploaded file: {str(e)}")
else:
    st.info("Please upload an image to begin prediction.")
