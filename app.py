import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from streamlit_drawable_canvas import st_canvas

# Load the trained model
model = tf.keras.models.load_model("digit_recognition_model.h5")
#Streamlit app title and description
st.title("Handwritten Digit Recognition")
st.write("Draw a digit in the box below, and I'll tell you what it is!")

# Create a canvas for drawing
canvas_result = st_canvas(
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas",
)

# Predict on drawing
if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Resize to 28x28, the input size for most digit recognition models
        img = Image.fromarray(canvas_result.image_data.astype("uint8"))
        img = img.convert("L")  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to 28x28 pixels
        
        img = np.array(img).reshape(1, 28, 28, 1) / 255.0  # Scale values
        
        # Prediction
        prediction = model.predict(img)
        st.write("Predicted Digit:", np.argmax(prediction))
