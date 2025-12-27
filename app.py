import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load the trained MNIST model
model = load_model("mnist_model.h5")

st.title("MNIST Digit Recognition Dashboard")
st.write("Draw a digit or upload an image to see the model prediction.")

# Sidebar option to choose input mode
mode = st.sidebar.selectbox("Choose input method", ["Draw Digit", "Upload Image"])

def preprocess_image(img):
    """Convert image to 28x28 grayscale and normalize"""
    img = img.convert("L").resize((28,28))  # Convert to grayscale & resize
    img_array = np.array(img)/255.0          # Normalize to 0-1
    img_array = 1 - img_array                # Invert colors: white background -> black
    img_array = img_array.reshape(1,28,28)  # Reshape for model
    return img_array

if mode == "Draw Digit":
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype('uint8'))
        img_array = preprocess_image(img)
        pred = model.predict(img_array)
        st.write(f"Predicted Digit: {np.argmax(pred)}")
        st.image(img.resize((140,140)), caption="Processed Image", width=140)

elif mode == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["png","jpg","jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img_array = preprocess_image(img)
        pred = model.predict(img_array)
        st.write(f"Predicted Digit: {np.argmax(pred)}")
        st.image(img.resize((140,140)), caption="Uploaded Image", width=140)
