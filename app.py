import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from precaution import precautions_page
from about import about_page
# Suppress warnings
import warnings

warnings.filterwarnings('ignore')

# Load the model
model = tf.keras.models.load_model('model.h5')

# Define class labels
class_labels = ['Non-demented', 'Very Mild Demented', 'Mild Demented', 'Moderate Demented']


def load_image(uploaded_file):
    image = Image.open(uploaded_file)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)

def preprocess_image(image_np):

    if len(image_np.shape) == 3:
        image_np = np.expand_dims(image_np, axis=0)
    try:
        resized_image = tf.image.resize(image_np, (128, 128), method='bicubic')
    except ValueError as e:
        st.error(
            f"Error resizing image: {e}. Please ensure the image has 3 dimensions (RGB) or 4 dimensions (batch + RGB).")
        return None

    resized_image = resized_image / 255.0  # Normalize
    return resized_image


def predict_image(image):
    prediction = model.predict(image)
    return class_labels[np.argmax(prediction)]


def classify_image():
    st.title("Classify Image")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = load_image(uploaded_file)
        st.image(Image.fromarray(image), caption='Uploaded Image', use_column_width=True)

        image_np = preprocess_image(image)
        if image_np is None:
            return

        if st.button("Classify"):
            st.write("Classification in progress...")
            prediction = predict_image(image_np)
            st.title(f"Predicted class: {prediction}")

def main():
    st.title("Alzheimer's Disease Image Classifier")
    page = st.sidebar.radio("Menu", ["Classify Image", "Precautions", "About"])

    if page == "Classify Image":
        classify_image()
    elif page == "Precautions":
        precautions_page()  # Call the function directly without using the module name
    elif page == "About":
        about_page()


if __name__ == "__main__":
    main()