import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the saved model
model_path = r'C:\Users\Administrator\Desktop\Miniproj\venv\fish\best\cnn_fish_rmsprop_0.2.h5'
model = tf.keras.models.load_model(model_path)

# Define class labels
class_labels_dict = {
    0: 'animal fish',
    1: 'animal fish bass',
    2: 'fish sea_food black_sea_sprat',
    3: 'fish sea_food gilt_head_bream',
    4: 'fish sea_food hourse_mackerel',
    5: 'fish sea_food red_mullet',
    6: 'fish sea_food red_sea_bream',
    7: 'fish sea_food sea_bass',
    8: 'fish sea_food shrimp',
    9: 'fish sea_food striped_red_mullet',
    10: 'fish sea_food trout'
}

# Define the Streamlit app
st.title("Fish Category Predictor")

# Upload image
uploaded_image = st.file_uploader("Choose a fish image", type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img_array = tf.keras.preprocessing.image.img_to_array(image.resize((256, 256)))
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    confidence_score = np.max(predictions, axis=1)

    # Display the results
    st.write(f"Predicted Category: {class_labels_dict.get(predicted_class[0], 'Unknown')}")
    st.write(f"Confidence Score: {confidence_score[0]:.2f}")