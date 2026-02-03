import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os
import time
# 👇 FIX 1: Import the correct preprocessing function
from tensorflow.keras.applications.efficientnet import preprocess_input

# 1. SETUP
st.set_page_config(page_title="V-Classify | Teacher Mode", page_icon="🚗")

MISTAKE_FOLDER = "hard_examples"
if not os.path.exists(MISTAKE_FOLDER):
    os.makedirs(MISTAKE_FOLDER)

@st.cache_resource
def load_model():
    # 👇 FIX 2: Load the new .keras file (not the old .h5)
    return tf.keras.models.load_model('best_vehicle_model.keras')

try:
    model = load_model()
except OSError:
    st.error("Model not found! Make sure 'best_vehicle_model.keras' is in the folder.")
    st.stop()

class_names = ['Bus', 'Hatchback', 'Pick-up', 'SUV', 'Sedan']

# 2. HELPER FUNCTIONS
def process_image(img, target_size=(224, 224)):
    # Standardize image size
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    background = Image.new('RGB', target_size, (0, 0, 0))
    offset = ((target_size[0] - img.size[0]) // 2, (target_size[1] - img.size[1]) // 2)
    background.paste(img, offset)
    return background

def save_mistake(image, true_label):
    class_folder = os.path.join(MISTAKE_FOLDER, true_label)
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)

    timestamp = int(time.time())
    unique_id = os.urandom(4).hex()
    filename = f"{timestamp}_{unique_id}.jpg"
    path = os.path.join(class_folder, filename)
    image.save(path)
    return path

# 3. UI
st.title("🚗 V-Classify: Batch Processing")
st.info("Upload up to 5 images. Scroll down to check each one.")

uploaded_files = st.file_uploader(
    "Choose images...", 
    type=["jpg", "png", "jpeg"], 
    accept_multiple_files=True 
)

if uploaded_files:
    if len(uploaded_files) > 5:
        st.warning("⚠️ You uploaded more than 5 files. Only processing the first 5.")
        uploaded_files = uploaded_files[:5]

    for i, uploaded_file in enumerate(uploaded_files):
        st.divider()
        st.write(f"### 🖼️ Image {i+1}: {uploaded_file.name}")
        
        original_image = Image.open(uploaded_file).convert('RGB')
        processed_image = process_image(original_image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(original_image, caption="Original", use_container_width=True)
        
        # 👇 FIX 3: Correct Preprocessing Pipeline
        # Convert to array (Values are 0-255)
        img_array = tf.keras.preprocessing.image.img_to_array(processed_image)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply EfficientNet specific preprocessing (No manual / 255.0 here!)
        img_array = preprocess_input(img_array)

        # Predict
        predictions = model.predict(img_array)
        
        predicted_index = int(np.argmax(predictions))
        predicted_class = class_names[predicted_index]
        confidence = 100 * np.max(predictions)

        with col2:
            st.metric("AI Prediction", f"{predicted_class}", f"{confidence:.1f}% Conf.")
            st.bar_chart(dict(zip(class_names, predictions[0])))

        with st.expander(f"Review & Teach Image {i+1}"):
            st.write("Is the AI wrong? Correct it below:")
            
            true_label = st.selectbox(
                f"Actual Class for Image {i+1}", 
                class_names, 
                index=predicted_index,
                key=f"select_{i}"
            )
            
            if st.button(f"Save Image {i+1} to Training Set", key=f"btn_{i}"):
                if true_label == predicted_class:
                    st.warning("AI was already correct.")
                else:
                    save_path = save_mistake(processed_image, true_label)
                    st.success(f"✅ Saved as {true_label}!")