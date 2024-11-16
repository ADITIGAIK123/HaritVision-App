import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image


# Set Page Title
st.set_page_config(page_title="HaritVision", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Select Section:", ["Home", "Model Info", "About"])

# Placeholder for dynamic content
if menu == "Home":
    st.title("Disease Detection App")
    st.write("Upload an apple tree image to predict its health status.")
elif menu == "Model Info":
    st.title("Model Information")
    st.write("This is a convolutional neural network trained on various disease categories.")
elif menu == "About":
    st.title("About HaritVision")
    st.write("HaritVision is an intelligent system for apple orchard management.")

if menu == "Home":
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Load Model
        model = tf.keras.models.load_model('path/to/haritvision_model.h5')

        # Preprocess Image
        img_array = image.resize((128, 128))  # Resize to match model input
        img_array = np.array(img_array) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict
        predictions = model.predict(img_array)
        class_labels = ["Healthy", "Apple Scab", "Cedar Apple Rust", "Alternaria Leaf Spot"]
        predicted_class = class_labels[np.argmax(predictions)]
        confidence = np.max(predictions)

        # Display Prediction
        st.write(f"Prediction: **{predicted_class}**")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")


# Load the trained model
model_path = '/content/drive/MyDrive/haritvision_model.keras'
model = tf.keras.models.load_model(model_path)

# Class labels for prediction mapping
class_labels = ["Alternaria leaf spot", "Apple scab", "Brown spot", "Cedar apple rust", "Gray spot", "Healthy tree"]

# Streamlit App Title
st.title("HaritVision: Apple Disease Detection System")

# Upload image
uploaded_file = st.file_uploader("Upload an image of an apple tree leaf", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image_data.resize((128, 128))  # Resize to match model's input size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]

    # Display results
    st.success(f"Predicted Class: {predicted_class}")
    st.write("Prediction Probabilities:")
    for i, label in enumerate(class_labels):
        st.write(f"{label}: {predictions[0][i]:.2%}")

import matplotlib.pyplot as plt

if menu == "Model Info":
    st.subheader("Model Training Performance")
    # Example data
    history_data = {
        "accuracy": [0.85, 0.88, 0.90],
        "val_accuracy": [0.82, 0.84, 0.86],
    }

    # Plot training performance
    fig, ax = plt.subplots()
    ax.plot(history_data['accuracy'], label='Training Accuracy')
    ax.plot(history_data['val_accuracy'], label='Validation Accuracy')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    st.pyplot(fig)
