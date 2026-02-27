import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="Crop Disease Classifier",
    page_icon="🌿",
    layout="centered",
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌿 Crop Disease Classifier")
st.write("Upload an image of a plant leaf to detect potential diseases.")

# Load model (Cache it)
@st.cache_resource
def load_trained_model():
    model_path = 'crop_disease_classifier_final.keras'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_trained_model()

# Class Names (PlantVillage)
# Note: In a real app, these should be loaded from a config file or tfds_info
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
    'Tomato___healthy'
]

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if model is None:
        st.error("Model file 'crop_disease_classifier_final.keras' not found. Please train the model first.")
    else:
        if st.button('Predict'):
            with st.spinner('Analyzing...'):
                # Preprocess
                img = image.resize((224, 224))
                img_array = np.array(img)
                img_array = np.expand_dims(img_array, axis=0)
                
                # Predict
                predictions = model.predict(img_array)
                score = tf.nn.softmax(predictions[0])
                
                class_idx = np.argmax(predictions[0])
                confidence = 100 * np.max(predictions[0])
                
                st.success(f"### Result: {class_names[class_idx].replace('___', ' - ')}")
                st.info(f"**Confidence:** {confidence:.2f}%")
                
                # Show bars for top 3
                top_3_indices = np.argsort(predictions[0])[-3:][::-1]
                st.write("#### Top 3 Predictions:")
                for idx in top_3_indices:
                    st.write(f"- {class_names[idx].replace('___', ' - ')}: {100 * predictions[0][idx]:.2f}%")
                    st.progress(float(predictions[0][idx]))

st.divider()
st.info("💡 Tip: For best results, ensure the leaf is clearly visible and well-lit.")
