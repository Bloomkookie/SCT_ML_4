import streamlit as st
import cv2
import numpy as np
from skimage.feature import hog
import joblib
from PIL import Image
import tempfile
import os

# Load SVM model and label encoder
model = joblib.load("svm_gesture_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Set Streamlit page config
st.set_page_config(page_title="✋ Hand Gesture Recognition", layout="centered")
st.title("🤖 Hand Gesture Recognition")
st.markdown("Upload an image or use webcam to predict your hand gesture using a pre-trained SVM model.")

# Function to extract HOG features
def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (64, 64))
    features = hog(resized,
                   orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys',
                   visualize=False,
                   feature_vector=True)
    return features

# Function to predict gesture
def predict_gesture(image):
    features = extract_hog_features(image)
    prediction = model.predict([features])
    gesture_name = label_encoder.inverse_transform(prediction)[0]
    return gesture_name

# --- Upload Image ---
st.subheader("📤 Upload Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)
    img_np = np.array(img)

    if st.button("🔍 Predict Uploaded Image"):
        gesture = predict_gesture(img_np)
        st.success(f"✅ Predicted Gesture: **{gesture}**")

# --- Webcam ---
st.subheader("📸 Capture from Webcam")

if st.checkbox("Activate Webcam"):
    picture = st.camera_input("Take a picture")
    if picture is not None:
        img = Image.open(picture).convert("RGB")
        st.image(img, caption="Captured Image", width=300)
        img_np = np.array(img)

        if st.button("🔍 Predict Webcam Image"):
            gesture = predict_gesture(img_np)
            st.success(f"✅ Predicted Gesture: **{gesture}**")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, OpenCV, HOG, and SVM")
