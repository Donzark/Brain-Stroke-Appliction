# 🧠 AI Powered Stroke Detection App

This is a **Streamlit application** that uses a **ResNet-based Convolutional Neural Network (CNN)** to classify CT scan images into two categories: **Normal** or **Stroke**. The app is designed to assist in the preliminary detection of strokes using medical CT scan images.

## 🔥 Try the App  

You can access the **AI-Powered Stroke Detection App** by clicking the link below:  

🔗 [**Brain Stroke Detection App**](https://brain-stroke-appliction.streamlit.app/)  

This app uses **deep learning** to analyze Brain CT scan images and detect potential stroke indicators with high accuracy. Upload an image and get an instant prediction!  

## 🖼️ Sample Images for Testing  

To test the app, here are some 📂 [**Sample Test Images**](./Sample_test_images) available in the repository.  

Download and upload these images to the app for quick testing and evaluation!

## 🚀 Features

- **ResNet-based Model**:  
  The application uses a ResNet model fine-tuned on a CT scan dataset.

- **Image Classification**:  
  Detects whether a CT scan image indicates a **normal brain** or signs of a **stroke**.
  
- **High Accuracy**:  
  The ResNet model achieved an impressive **94% accuracy** on the test dataset, demonstrating its reliability for stroke detection.

- **Interactive Interface**:  
  Built with Streamlit, the app provides a user-friendly interface for image upload and classification.

- **File Support**:  
  Accepts `.jpg`, `.jpeg`, and `.png` image formats.

---

## 🛠️ Installation

Follow these steps to set up and run the application locally:

### 1. Clone the Repository
```bash
git clone https://github.com/donzark/stroke-detection-app.git
cd stroke-detection-app
```

### 2. Install Dependencies
Install the required Python libraries using pip:

```bash
pip install -r requirements.txt
```

### 3. Add the Model
Save your trained ResNet-based model (e.g., resnet_model.h5) in the root directory or provide its path in the code.

### 4. Run the Application
Start the Streamlit app:

```bash
streamlit run stroke_detection_app.py
```

## 📂 Project Structure
```graphql
stroke-detection-app/
│
├── stroke_detection_app.py    # Streamlit application code
├── resnet_sigmoid_model       # Trained ResNet model 
├── requirements.txt           # List of dependencies
├── README.md                  # Project README file
└── sample_images/             # Sample CT scan images for testing 
```

## ⚙️ How to Use

1. **Upload an Image**:  
   Click the "Upload" button and select a CT scan image (`.jpg`, `.jpeg`, or `.png`).

2. **Predict**:  
   Click the "Predict" button to classify the image as **Normal** or **Stroke**.

3. **View Results**:  
   The app displays:
   - The predicted class (e.g., **Stroke** or **Normal**).
   - The model’s confidence score.
   - A bar chart showing confidence percentages for all classes.


## 🔧 Requirements

- **Python 3.8+**

- **Dependencies** (stored in `requirements.txt`):
  - `streamlit`
  - `tensorflow`
  - `altair`
  - `numpy`
  - `pandas`
  - `Pillow`

To install all dependencies, run:
```bash
pip install -r requirements.txt
```


## 📂 Dataset

This application uses the **[Brain Stroke CT Image Dataset](https://www.kaggle.com/datasets/afridirahman/brain-stroke-ct-image-dataset/data)**, a collection of CT scan images categorized into two classes:  
1. **Normal**: Images of CT scans without signs of a stroke.  
2. **Stroke**: Images of CT scans showing signs of a stroke.

### **Dataset Details**
- **Source**: [Kaggle](https://www.kaggle.com/datasets/afridirahman/brain-stroke-ct-image-dataset/data)
- **Images**: 1900 CT scan images:
  - **Normal**: 950 images
  - **Stroke**: 950 images
- **Image Dimensions**: Images are in various resolutions but resized to **224x224** pixels during preprocessing for compatibility with the model.
- **File Formats**: `.png`

### **Preprocessing Steps**
1. **Resizing**: All images are resized to `(224, 224)` pixels to match the ResNet model's input size.
2. **Channel Conversion**: Images are converted to RGB to ensure compatibility with the pre-trained ResNet architecture.
3. **Normalization**: Pixel values are normalized to the range `[0, 1]` to improve model training performance.

### **Dataset Usage**
- **Training Data**: 70% of the dataset (split using `train_test_split`) is used for model training.
- **Test Data**: 30% of the dataset is reserved for evaluating the model's performance.

## ✨ Future Improvements

- Add more detailed stroke classification (e.g., ischemic vs. hemorrhagic).
- Incorporate additional datasets for improved accuracy.
- Deploy the app to a cloud platform (e.g., Streamlit Sharing, AWS, or Heroku).

## 🙌 Acknowledgements

This project was proudly sponsored by **NITDA (National Information Technology Development Agency)** through the **Nigeria Artificial Intelligence Research Scheme (NAIRS)**.

### **About NITDA**
NITDA is committed to empowering the future through innovation and digital transformation, supporting groundbreaking research in artificial intelligence and related fields.

### **Grant Details**
- **Recipient**: Dr. Obasa, Adekunle Isiaka (Nigerian Researcher)  
- **Research Focus**: Application of hybrid machine learning techniques to improve the diagnosis of ischemic strokes in CT scan images.  

We extend our gratitude to **NITDA** and **NAIRS** for supporting this initiative and contributing to advancements in healthcare technology.

## 🤝 Contributing
Contributions are welcome! Feel free to submit a pull request or open an issue to suggest improvements.

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.





