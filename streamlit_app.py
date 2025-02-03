# import os
# os.system("pip install tensorflow-hub")
# import sys
# sys.path.append("/home/appuser/.local/lib/python3.12/site-packages")
# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
# import numpy as np
# import pandas as pd
# import altair as alt
# import tensorflow_hub as hub
# from tensorflow.keras.models import load_model
# import tf_keras

# # Streamlit Layout
# st.set_page_config(page_title="Stroke Detection", page_icon="🧠")

# # Function to preprocess and predict
# @st.cache_resource
# def predict_image(image_path, model):
#     import tensorflow as tf
#     import numpy as np
#     from tensorflow.keras.preprocessing.image import load_img, img_to_array

#     # Load and preprocess image
#     image = load_img(image_path, target_size=(224, 224))
#     image = img_to_array(image)
#     image = np.expand_dims(image, axis=0)  # Add batch dimension

#     # Make prediction
#     preds = model(image)
    
#     # Check if preds is a dictionary
#     if isinstance(preds, dict):
#         key = list(preds.keys())[0]  # Get first key
#         preds = preds[key]  # Extract tensor
    
#     preds = preds.numpy() if hasattr(preds, "numpy") else np.array(preds)

#     # Convert probability to class
#     pred_class = "Potential Stroke Detected. Immediate medical evaluation needed." if preds[0] > 0.5 else "No Stroke Indicators Detected."
#     pred_conf = preds[0]  # Confidence score

#     return pred_class, pred_conf

# # Define class names
# class_names = ["Normal", "Stroke"]  # Update based on your dataset's classes

# # Sidebar
# # st.sidebar.title("Stroke Detection App")
# # st.sidebar.write("""
# # This application uses a **ResNet-based CNN model** to detect whether a CT scan image indicates a **Normal** brain or signs of a **Stroke**.

# # - **Model:** ResNet
# # - **Dataset:** Custom CT Scan Dataset
# # """)

# # Sidebar
# st.sidebar.title("🧠 Stroke Detection App")
# st.sidebar.markdown("""
# This application uses an advanced **ResNet-based CNN model** to analyze Brain CT scan images and detect possible signs of **Stroke**.

# ### 🔍 Features:
# - **Deep Learning Model:** ResNet-based CNN
# - **Dataset:** Custom CT Scan Dataset
# - **Accuracy:** High-performance stroke detection with 94% Accuracy
# - **Fast & Secure:** AI-powered real-time diagnosis
# """)

# # Main App
# # st.title("🧠 Stroke Detection")
# # st.write("Upload a Brain CT scan image to detect signs of a stroke.")
# st.title("🧠 Stroke Detection Using AI")
# st.markdown("Upload a **Brain CT scan image** to detect signs of a stroke.")

# # File uploader
# uploaded_file = st.file_uploader("Upload a Brain CT scan image (JPG, PNG, or JPEG)", type=["jpg", "jpeg", "png"])

# from tensorflow.keras.layers import TFSMLayer

# MODEL_PATH = "./resnet_sigmoid_model"
# model = tf.keras.layers.TFSMLayer(MODEL_PATH, call_endpoint='serving_default') 

# if uploaded_file is not None:
#     # Display the uploaded image
#     st.image(uploaded_file, caption="Uploaded Image", width=300)
#     pred_button = st.button("Predict")

#     if pred_button:
#         # Save the uploaded file temporarily
#         with open("temp_image.jpg", "wb") as f:
#             f.write(uploaded_file.getbuffer())

#         # Predict using the model
#         pred_class, pred_conf = predict_image("temp_image.jpg", model)

#         # Display results
#         # Adjust confidence to always show high values
#         if pred_class == "No Stroke Indicators Detected.":
#             adjusted_conf = 100 - (pred_conf[0] * 100)
#         else:
#             adjusted_conf = pred_conf[0] * 100
            
#         # Print the formatted confidence
#         st.success(f"Prediction: **{pred_class}**\n\nConfidence: **{adjusted_conf:.2f}%**")

# else:
#     st.warning("Please upload a **Brain CT scan image** to proceed.")


# import os
# os.system("pip install tensorflow-hub")
# import sys
# sys.path.append("/home/appuser/.local/lib/python3.12/site-packages")
# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
# import numpy as np
# import pandas as pd
# import altair as alt
# import tensorflow_hub as hub
# from tensorflow.keras.models import load_model
# import tf_keras

# # Streamlit Layout
# st.set_page_config(
#     page_title="Stroke Detection",
#     page_icon="🧠",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Function to preprocess and predict
# @st.cache_resource
# def predict_image(image_path, model):
#     import tensorflow as tf
#     import numpy as np
#     from tensorflow.keras.preprocessing.image import load_img, img_to_array

#     # Load and preprocess image
#     image = load_img(image_path, target_size=(224, 224))
#     image = img_to_array(image)
#     image = np.expand_dims(image, axis=0)  # Add batch dimension

#     # Make prediction
#     preds = model(image)
    
#     # Check if preds is a dictionary
#     if isinstance(preds, dict):
#         key = list(preds.keys())[0]  # Get first key
#         preds = preds[key]  # Extract tensor
    
#     preds = preds.numpy() if hasattr(preds, "numpy") else np.array(preds)

#     # Convert probability to class
#     pred_class = "⚠️ Potential Stroke Detected. Immediate medical evaluation needed." if preds[0] > 0.5 else "✅ No Stroke Indicators Detected."
#     pred_conf = preds[0]  # Confidence score

#     return pred_class, pred_conf

# # Define class names
# class_names = ["Normal", "Stroke"]

# # Sidebar
# with st.sidebar:
#     st.title("🧠 Stroke Detection App")
#     st.markdown("""
#     This application uses an advanced **ResNet-based CNN model** to analyze Brain CT scan images and detect possible signs of **Stroke**.
    
#     ### 🔍 Features:
#     - **Deep Learning Model:** ResNet-based CNN
#     - **Dataset:** Custom CT Scan Dataset
#     - **Accuracy:** High-performance stroke detection with **94% Accuracy**
#     - **Fast & Secure:** AI-powered real-time diagnosis
#     """)
#     st.markdown("---")
#     st.info("💡 Upload a Brain CT scan image on the right panel to get a stroke detection diagnosis.")

#     # Acknowledgment Section
#     st.markdown("---")
#     st.markdown("### 🙌 Acknowledgment")
#     st.markdown("""
#     This project is supported by **NITDA** through the **Nigeria AI Research Scheme (NAIRS)**.  
#     Special recognition to **Dr. Obasa, Adekunle Isiaka** for pioneering this research.
#     """)


# # Main App Layout
# st.title("🧠 AI-Powered Stroke Detection")
# st.markdown("Upload a **Brain CT scan image** to detect signs of a stroke.")

# # Create a container for better structure
# with st.container():
#     col1, col2 = st.columns([1, 2])

#     # File uploader on the left
#     with col1:
#         uploaded_file = st.file_uploader(
#             "📤 Upload a Brain CT scan image (JPG, PNG, or JPEG)", 
#             type=["jpg", "jpeg", "png"]
#         )

#     # Model loading and prediction logic
#     from tensorflow.keras.layers import TFSMLayer
#     MODEL_PATH = "./resnet_sigmoid_model"
#     model = tf.keras.layers.TFSMLayer(MODEL_PATH, call_endpoint='serving_default') 

#     if uploaded_file:
#         with col2:
#             st.image(uploaded_file, caption="📷 Uploaded Image", width=300)
        
#         pred_button = st.button("🧠 Analyze Image")

#         if pred_button:
#             with st.spinner("🔄 Analyzing the CT scan..."):
#                 # Save the uploaded file temporarily
#                 with open("temp_image.jpg", "wb") as f:
#                     f.write(uploaded_file.getbuffer())

#                 # Predict using the model
#                 pred_class, pred_conf = predict_image("temp_image.jpg", model)

#                 # Adjust confidence to always show high values
#                 adjusted_conf = 100 - (pred_conf[0] * 100) if "No Stroke" in pred_class else pred_conf[0] * 100

#                 # Display Results
#                 if "No Stroke" in pred_class:
#                     st.success(f"✅ **{pred_class}**\n\nConfidence: **{adjusted_conf:.2f}%**")
#                 else:
#                     st.error(f"⚠️ **{pred_class}**\n\nConfidence: **{adjusted_conf:.2f}%**")

#     else:
#         st.warning("⚠️ Please upload a **Brain CT scan image** to proceed.")

# import os
# os.system("pip install tensorflow-hub")
# import sys
# sys.path.append("/home/appuser/.local/lib/python3.12/site-packages")
# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import img_to_array, load_img
# import numpy as np
# import tensorflow_hub as hub
# from tensorflow.keras.models import load_model
# import tf_keras

# # Streamlit Layout
# st.set_page_config(
#     page_title="Stroke Detection",
#     page_icon="🧠",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Function to preprocess and predict
# @st.cache_resource
# def predict_image(image_path, model):
#     import tensorflow as tf
#     import numpy as np
#     from tensorflow.keras.preprocessing.image import load_img, img_to_array

#     # Load and preprocess image
#     image = load_img(image_path, target_size=(224, 224))
#     image = img_to_array(image)
#     image = np.expand_dims(image, axis=0)  # Add batch dimension

#     # Make prediction
#     preds = model(image)
    
#     # Check if preds is a dictionary
#     if isinstance(preds, dict):
#         key = list(preds.keys())[0]  # Get first key
#         preds = preds[key]  # Extract tensor
    
#     preds = preds.numpy() if hasattr(preds, "numpy") else np.array(preds)

#     # Convert probability to class
#     pred_class = " Potential Stroke Detected. Immediate medical evaluation is Adviced!." if preds[0] > 0.5 else " No Stroke Indicators Detected."
#     pred_conf = preds[0]  # Confidence score

#     return pred_class, pred_conf

# # Sidebar
# with st.sidebar:
#     st.title("🧠 Stroke Detection App")
#     st.markdown("""
#     This application uses an advanced **ResNet-based CNN model** to analyze Brain CT scan images and detect possible signs of **Stroke**.
    
#     ### 🔍 Features:
#     - **Deep Learning Model:** ResNet-based CNN
#     - **Dataset:** Custom CT Scan Dataset
#     - **Accuracy:** High-performance stroke detection with **94% Accuracy**
#     - **Fast & Secure:** AI-powered real-time diagnosis
#     """)
#     st.markdown("---")
#     st.info("💡 Upload a Brain CT scan image on the right panel to get a stroke detection diagnosis.")

#      # Acknowledgment Section
#     st.markdown("---")
#     st.markdown("### 🙌 Acknowledgment")
#     st.markdown("""
#     This project is supported by **NITDA** through the **Nigeria AI Research Scheme (NAIRS)**.  
#     Special recognition to **Dr. Obasa, Adekunle Isiaka** for pioneering this research.
#     """)

# # Store session state
# if "uploaded_file" not in st.session_state:
#     st.session_state.uploaded_file = None
# if "prediction" not in st.session_state:
#     st.session_state.prediction = None

# # Centered Layout Before Upload
# st.title("🧠 AI-Powered Stroke Detection")
# st.markdown("Upload a **Brain CT scan image** to detect signs of a stroke.")

# uploaded_file = st.file_uploader(
#     "📤 Upload a Brain CT scan image (JPG, PNG, or JPEG)", 
#     type=["jpg", "jpeg", "png"]
# )

# if uploaded_file:
#     st.session_state.uploaded_file = uploaded_file  # Save to session state

# # After Upload: Switch to Two-Column Layout
# if st.session_state.uploaded_file:
#     col1, col2 = st.columns([1, 2])

#     with col1:
#         st.image(st.session_state.uploaded_file, caption="📷 Uploaded Image", width=300)

#     with col2:
#         pred_button = st.button("🧠 Analyze Image")

#         if pred_button:
#             with st.spinner("🔄 Analyzing the CT scan..."):
#                 # Save the uploaded file temporarily
#                 with open("temp_image.jpg", "wb") as f:
#                     f.write(st.session_state.uploaded_file.getbuffer())

#                 # Model loading and prediction logic
#                 from tensorflow.keras.layers import TFSMLayer
#                 MODEL_PATH = "./resnet_sigmoid_model"
#                 model = tf.keras.layers.TFSMLayer(MODEL_PATH, call_endpoint='serving_default')

#                 # Predict using the model
#                 pred_class, pred_conf = predict_image("temp_image.jpg", model)

#                 # Adjust confidence to always show high values
#                 adjusted_conf = 100 - (pred_conf[0] * 100) if "No Stroke" in pred_class else pred_conf[0] * 100

#                 # Save prediction to session state
#                 st.session_state.prediction = (pred_class, adjusted_conf)

# # Display Prediction (Persists after rerun)
# if st.session_state.prediction:
#     pred_class, adjusted_conf = st.session_state.prediction

#     if "No Stroke" in pred_class:
#         st.success(f"✅ **{pred_class}**\n\nConfidence: **{adjusted_conf:.2f}%**")
#     else:
#         st.error(f"⚠️ **{pred_class}**\n\nConfidence: **{adjusted_conf:.2f}%**")


# else:
#     st.warning("⚠️ Please upload a **Brain CT scan image** to proceed.")

import os
os.system("pip install tensorflow-hub")
import sys
sys.path.append("/home/appuser/.local/lib/python3.12/site-packages")

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
import tf_keras

# Streamlit Layout
st.set_page_config(
    page_title="Stroke Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to preprocess and predict
@st.cache_resource
def load_prediction_model():
    """Loads the trained model and caches it."""
    MODEL_PATH = "./resnet_sigmoid_model"
    model = tf.keras.layers.TFSMLayer(MODEL_PATH, call_endpoint='serving_default')
    return model

model = load_prediction_model()

@st.cache_resource
def predict_image(image_path, model):
    """Processes image and makes predictions."""
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    preds = model(image)
    
    if isinstance(preds, dict):  
        key = list(preds.keys())[0]  
        preds = preds[key]  
    
    preds = preds.numpy() if hasattr(preds, "numpy") else np.array(preds)

    pred_class = "⚠️ Potential Stroke Detected. Immediate medical evaluation needed." if preds[0] > 0.5 else "✅ No Stroke Indicators Detected."
    pred_conf = preds[0]  

    return pred_class, pred_conf

# Sidebar
with st.sidebar:
    st.title("🧠 Stroke Detection App")
    st.markdown("""
    This application uses an advanced **ResNet-based CNN model** to analyze Brain CT scan images and detect possible signs of **Stroke**.
    
    ### 🔍 Features:
    - **Deep Learning Model:** ResNet-based CNN
    - **Dataset:** Custom CT Scan Dataset
    - **Accuracy:** High-performance stroke detection with **94% Accuracy**
    - **Fast & Secure:** AI-powered real-time diagnosis
    """)
    st.markdown("---")
    st.info("💡 Upload a Brain CT scan image on the right panel to get a stroke detection diagnosis.")
    st.markdown("---")
    st.markdown("### 🙌 Acknowledgment")
    st.markdown("""
    This project is supported by **NITDA** through the **Nigeria AI Research Scheme (NAIRS)**.  
    Special recognition to **Dr. Obasa, Adekunle Isiaka** for pioneering this research.
    """)

# Title (Centered)
st.title("🧠 AI-Powered Stroke Detection")
st.markdown("Upload a **Brain CT scan image** to detect signs of a stroke.")

# Session State to Handle File Persistence
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# File Uploader (Centered)
uploaded_file = st.file_uploader(
    "📤 Upload a Brain CT scan image (JPG, PNG, or JPEG)", 
    type=["jpg", "jpeg", "png"]
)

# Store in session state
if uploaded_file:
    st.session_state.uploaded_file = uploaded_file

# After Upload: Two-Column Layout
if st.session_state.uploaded_file:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(st.session_state.uploaded_file, caption="📷 Uploaded Image", width=300)

    with col2:
        pred_button = st.button("🧠 Analyze Image")

        if pred_button:
            with st.spinner("🔄 Analyzing the CT scan..."):
                with open("temp_image.jpg", "wb") as f:
                    f.write(st.session_state.uploaded_file.getbuffer())

                pred_class, pred_conf = predict_image("temp_image.jpg", model)

                adjusted_conf = 100 - (pred_conf[0] * 100) if "No Stroke" in pred_class else pred_conf[0] * 100

                st.session_state.prediction = (pred_class, adjusted_conf)

# Display Prediction (Persists after rerun)
if st.session_state.prediction:
    pred_class, adjusted_conf = st.session_state.prediction

    with col2:
        if "No Stroke" in pred_class:
            st.success(f"✅ **{pred_class}**\n\nConfidence: **{adjusted_conf:.2f}%**")
        else:
            st.error(f"⚠️ **{pred_class}**\n\nConfidence: **{adjusted_conf:.2f}%**")

# Warning Message (Only Show if No Image is Uploaded)
if not st.session_state.uploaded_file:
    st.warning("⚠️ Please upload a **Brain CT scan image** to proceed.")

