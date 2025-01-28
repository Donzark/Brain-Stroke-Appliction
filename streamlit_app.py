import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import pandas as pd
import altair as alt

# Streamlit Layout
st.set_page_config(page_title="Stroke Detection", page_icon="🧠")

# Sidebar
st.sidebar.title("Stroke Detection App")
st.sidebar.write("""
This application uses a **ResNet-based CNN model** to detect whether a CT scan image indicates a **Normal** brain or signs of a **Stroke**.

- **Model:** ResNet
- **Dataset:** Custom CT Scan Dataset
""")

# Main App
st.title("🧠 Stroke Detection")
st.write("Upload a CT scan image to detect signs of a stroke.")

# File uploader
uploaded_file = st.file_uploader("Upload a CT scan image (JPG, PNG, or JPEG)", type=["jpg", "jpeg", "png"])

