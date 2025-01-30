import os
os.system("pip install tensorflow-hub")
import sys
sys.path.append("/home/appuser/.local/lib/python3.12/site-packages")
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import pandas as pd
import altair as alt
import tensorflow_hub as hub
import tf_keras

# Streamlit Layout
st.set_page_config(page_title="Stroke Detection", page_icon="🧠")

# @st.cache_resource
# def predict_image(image, _model):
#     # Load and preprocess the image
#     image = load_img(image, target_size=(224, 224))  # Resize to model's input size
#     image = img_to_array(image) / 255.0  # Normalize pixel values
#     image = np.expand_dims(image, axis=0)  # Add batch dimension

#     # Make predictions
#     preds = model.predict(image)

#     # Debugging: Show raw predictions
#     st.subheader("🔍 Debugging Information.")
#     st.write("**Raw model outputs:**", preds)
#     st.write("**Predicted class index:**", int(np.argmax(preds[0])))
#     st.write("**Predicted confidence:**", float(np.max(preds[0])))

#     # Extract class name and confidence
#     pred_class = class_names[np.argmax(preds[0])]
#     pred_conf = np.max(preds[0])

#     # Create a DataFrame for visualization
#     df = pd.DataFrame({
#         "Class": class_names,
#         "Confidence (%)": preds[0] * 100,
#         "color": ['#EC5953' if i == np.argmax(preds[0]) else '#3498DB' for i in range(len(class_names))]
#     })
#     df = df.sort_values("Confidence (%)", ascending=False)

#     # Debugging: Show sorted DataFrame in Streamlit
#     st.write("**Prediction Breakdown:**")
#     st.dataframe(df)

#     return pred_class, pred_conf, df


# # Function to preprocess and predict
# @st.cache_resource #(suppress_st_warning=True)
# def predict_image(image, _model):
#     # Load and preprocess the image
#     image = load_img(image, target_size=(224, 224))  # Resize to model's input size
#     image = img_to_array(image)/255.0  # Normalize pixel values
#     image = np.expand_dims(image, axis=0)  # Add batch dimension

#     # Make predictions
#     preds = model.predict(image)
#     pred_class = "Potential Stroke Detected. Immediate medical evaluation is advised!" if preds[0] > 0.5 else "No Stroke Indicators Detected."

#     pred_conf = float(preds[0]) 

#     # Create a DataFrame for visualization
#     df = pd.DataFrame({
#         "Class": class_names,
#         "Confidence (%)": preds[0] * 100,
#         "color": ['#EC5953' if i == np.argmax(preds[0]) else '#3498DB' for i in range(len(class_names))]
#     })
#     df = df.sort_values("Confidence (%)", ascending=False)
#     return pred_class, pred_conf, df


# Function to preprocess and predict
@st.cache_resource
def predict_image(image, _model):
    # Load and preprocess the image
    image = load_img(image, target_size=(224, 224))  # Resize to model's input size
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    preds = model.predict(image)[0][0]  # Extract single probability value

    # Define class names
    class_names = ["Healthy", "Stroke"]

    # Convert probability to percentages
    stroke_confidence = preds * 100
    healthy_confidence = 100 - stroke_confidence

    # Determine prediction class and display confidence
    if preds > 0.5:
        pred_class = "Potential Stroke Detected. Immediate medical evaluation is advised!"
        pred_conf = stroke_confidence  # Confidence reflects Stroke prediction
    else:
        pred_class = "No Stroke Indicators Detected."
        pred_conf = healthy_confidence  # Confidence reflects Healthy prediction

    # Create DataFrame for visualization
    df = pd.DataFrame({
        "Class": class_names,
        "Confidence (%)": [healthy_confidence, stroke_confidence],
        "color": ['#3498DB', '#EC5953']
    })

    return pred_class, pred_conf, df



# @st.cache_resource
# def predict_image(image, _model):
#     # Load and preprocess the image
#     image = load_img(image, target_size=(224, 224))  # Resize to model's input size
#     image = img_to_array(image) / 255.0  # Normalize pixel values
#     image = np.expand_dims(image, axis=0)  # Add batch dimension

#     # Make predictions
#     preds = model.predict(image)
    
#     print("Raw model outputs:", preds)  # Debugging step
#     print("Predicted class:", np.argmax(preds[0]))
#     print("Predicted confidence:", np.max(preds[0]))

#     pred_class = class_names[np.argmax(preds[0])]
#     pred_conf = np.max(preds[0])

#     # Create a DataFrame for visualization
#     df = pd.DataFrame({
#         "Class": class_names,
#         "Confidence (%)": preds[0] * 100,
#         "color": ['#EC5953' if i == np.argmax(preds[0]) else '#3498DB' for i in range(len(class_names))]
#     })
#     df = df.sort_values("Confidence (%)", ascending=False)
#     return pred_class, pred_conf, df


# Define class names
class_names = ["Normal", "Stroke"]  # Update based on your dataset's classes

# Sidebar
st.sidebar.title("Stroke Detection App")
st.sidebar.write("""
This application uses a **ResNet-based CNN model** to detect whether a CT scan image indicates a **Normal** brain or signs of a **Stroke**.

- **Model:** ResNet
- **Dataset:** Custom CT Scan Dataset
""")

# Main App
st.title("🧠 Stroke Detection")
st.write("Upload a Brain CT scan image to detect signs of a stroke.")

# File uploader
uploaded_file = st.file_uploader("Upload a CT scan image (JPG, PNG, or JPEG)", type=["jpg", "jpeg", "png"])

# Load the model
MODEL_PATH = "./resnet_sigmoid_model.h5"
model = tf_keras.models.load_model(MODEL_PATH, custom_objects={'KerasLayer': hub.KerasLayer})

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", width=300)
    pred_button = st.button("Predict")

    if pred_button:
        # Save the uploaded file temporarily
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Predict using the model
        pred_class, pred_conf, df = predict_image("temp_image.jpg", model)

        # Display results
        st.success(f"Prediction: **{pred_class}**\nConfidence: **{pred_conf * 100:.2f}%**")
        st.write(alt.Chart(df).mark_bar().encode(
            x='Confidence (%)',
            y=alt.Y('Class', sort=None),
            color=alt.Color("color", scale=None),
            text='Confidence (%)'
        ).properties(width=600, height=400).mark_text(align="left", dx=5))
else:
    st.warning("Please upload an image to proceed.")

