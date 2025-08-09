import streamlit as st
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

Model = tf.saved_model.load("models/1")
predict_fn = Model.signatures['serving_default']
Class_names = ["Early_blight", "Late_blight", "Healthy"]

def read_file_as_image(data) -> np.ndarray:
    try:
        return np.array(Image.open(BytesIO(data)))
    except Exception as e:
        st.error(f"Error reading image: {e}")
        return None

st.set_page_config(page_title="Potato Disease Classifier", page_icon="🥔", layout="centered")
st.title("🥔 Potato Leaf Disease Classification")
st.write("Upload an image of a potato leaf and get the prediction.")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        uploaded_file.seek(0)
        
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        uploaded_file.seek(0)
        img_array = read_file_as_image(uploaded_file.read())
        if img_array is not None:
            img_batch = np.expand_dims(img_array, 0)
            img_tensor = tf.constant(img_batch, dtype=tf.float32)
            
            try:
                predictions = predict_fn(img_tensor)
                output_key = list(predictions.keys())[0]
                predicted_class_index = np.argmax(predictions[output_key][0])
                predicted_class = Class_names[predicted_class_index]
                confidence = float(np.max(predictions[output_key][0]))
                
                st.subheader("Prediction Result")
                st.write(f"**Class:** {predicted_class}")
                st.write(f"**Confidence:** {confidence:.2%}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.error("Uploaded file is not a valid image.")
    except Exception as e:
        st.error(f"Error processing uploaded image: {e}")
