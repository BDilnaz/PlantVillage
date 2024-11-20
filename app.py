import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image

# Path to the saved model
MODEL_PATH = 'improved_model.h5'

# Load the model
model = load_model(MODEL_PATH)

# Constants
IMG_SIZE = (160, 160)  # Resize image to fit the model input

# Class indices (mapping the class names to the model output)
class_indices = {
    'Pepper__bell___Bacterial_spot': 0,
    'Pepper__bell___healthy': 1,
    'Potato___Early_blight': 2,
    'Potato___healthy': 3,
    'Potato___Late_blight': 4,
    'Tomato__Target_Spot': 5,
    'Tomato__Tomato_mosaic_virus': 6,
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 7,
    'Tomato_Bacterial_spot': 8,
    'Tomato_Early_blight': 9,
    'Tomato_healthy': 10,
    'Tomato_Late_blight': 11,
    'Tomato_Leaf_Mold': 12,
    'Tomato_Septoria_leaf_spot': 13,
    'Tomato_Spider_mites_Two_spotted_spider_mite': 14
}

# Streamlit app title
st.title("Leaf Disease Classification")
st.write("Upload an image of a leaf to classify its condition.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    image = image.resize(IMG_SIZE)  # Resize to match model input size
    image_array = img_to_array(image) / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make prediction
    st.write("Classifying the image...")
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=-1)[0]  # Get the class with the highest probability
    
    # Map the predicted class index to its label
    predicted_label = list(class_indices.keys())[predicted_class]
    confidence = prediction[0][predicted_class]

    # Display result
    st.write(f"**Prediction:** {predicted_label}")
    st.write(f"**Confidence:** {confidence:.2f}")

else:
    st.write("Please upload an image to classify.")
