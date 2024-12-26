import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the best model
MODEL_PATH = "best_model_cnn.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
class_labels = ['Parasitized', 'Uninfected']

# Function to preprocess the uploaded image
def preprocess_image(image, img_height=128, img_width=128):
    image = image.convert("RGB")
    image = image.resize((img_height, img_width))
    image = np.array(image)
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit application
st.title("Malaria Cell Image Classification")
st.write("Upload a cell image to classify it as **Parasitized** or **Uninfected**.")

# File uploader
uploaded_file = st.file_uploader("Choose a cell image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = class_labels[int(prediction[0] > 0.5)]
    confidence = prediction[0][0]
    # confidence = prediction[0][0] if predicted_class == 'Parasitized' else 1 - prediction[0][0]
    
    # Display the results
    st.write(f"### Prediction: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.2%}**")
