import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Try loading the model
try:
    model = load_model("traffic_sign_cnn_model.h5")
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

# Define class labels (Adjust based on your dataset)
class_labels = {
    0: "Speed Limit 20 km/h", 1: "Speed Limit 30 km/h", 2: "Speed Limit 50 km/h",
    3: "Speed Limit 60 km/h", 4: "Speed Limit 70 km/h", 5: "Speed Limit 80 km/h",
    6: "End of Speed Limit 80 km/h", 7: "Speed Limit 100 km/h", 8: "Speed Limit 120 km/h",
    9: "No Passing", 10: "No Passing for Vehicles over 3.5t", 11: "Right of Way at Intersection",
    12: "Priority Road", 13: "Yield", 14: "Stop", 15: "No Vehicles", 16: "Vehicles > 3.5t Prohibited",
    17: "No Entry", 18: "General Caution", 19: "Dangerous Curve Left", 20: "Dangerous Curve Right",
    21: "Double Curve", 22: "Bumpy Road", 23: "Slippery Road", 24: "Road Narrows on Right",
    25: "Road Work", 26: "Traffic Signals", 27: "Pedestrian Crossing", 28: "Children Crossing",
    29: "Bicycles Crossing", 30: "Beware of Ice/Snow", 31: "Wild Animals Crossing",
    32: "End of Speed and Passing Limits", 33: "Turn Right Ahead", 34: "Turn Left Ahead",
    35: "Ahead Only", 36: "Go Straight or Right", 37: "Go Straight or Left",
    38: "Keep Right", 39: "Keep Left", 40: "Roundabout Mandatory",
    41: "End of No Passing", 42: "End of No Passing for Vehicles > 3.5t"
}

# Basic UI
st.title("🚦 Traffic Sign Recognition AI")
st.write("Upload an image of a traffic sign, and the AI will recognize it!")

# Options to upload the file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Check if a file is uploaded
if uploaded_file is not None:
    try:
        # Load and preprocess the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 Uploaded Traffic Sign", use_column_width=True)
        st.write("🔄 Processing...")

        # Convert grayscale images to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize to match model input size
        image = image.resize((32, 32))

        # Convert to array and normalize
        image = img_to_array(image) / 255.0  
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction)
        confidence_score = np.max(prediction) * 100  # Get confidence percentage
        predicted_label = class_labels.get(predicted_class, "Unknown Sign")

        # Display prediction with confidence
        st.success(f"🛑 **Predicted Traffic Sign:** {predicted_label}")
        st.info(f"🎯 **Confidence Score:** {confidence_score:.2f}%")

    except Exception as e:
        st.error(f"⚠️ An error occurred while processing the image: {e}")
else:
    st.warning("⚠️ Please upload an image to proceed.")
