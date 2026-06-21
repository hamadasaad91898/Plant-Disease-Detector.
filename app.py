import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# 1. Page settings
st.set_page_config(page_title="Plant Disease AI", page_icon="🌿")

# Sidebar details for presentation
st.sidebar.title("🌿 Plant AI System")



# 2. Load the model and class names
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("plant_disease_model.keras", compile=False)
    return model

@st.cache_data
def load_class_names():
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    # Reverse dictionary (ID -> Name)
    class_names = {str(v): k for k, v in class_indices.items()}
    return class_names

model = load_model()
class_names = load_class_names()

# 3. Main UI
st.title("🌱 Plant Disease Prediction")
st.write("Upload a leaf image to predict the disease.")

# 4. Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)
    
    if st.button("Predict"):
        st.write("Analyzing...")
        
        # 5. Preprocess the image
        img = image.resize((224, 224))                  # Resize
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)   # Add batch
        img_array = img_array.astype('float32') / 255.0 # Normalize
        
        # 6. Predict
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        confidence = np.max(predictions) * 100
        
        # 7. Show result
        result_name = class_names[str(predicted_index)]
        
        st.success(f"Prediction: {result_name.upper()}")
        st.info(f"Confidence: {confidence:.2f}%")
