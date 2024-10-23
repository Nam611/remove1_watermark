import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import io

# Load your model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('watermark_auto_encoder_model.h5', compile=False)
    model.compile(optimizer='adam', loss='mean_squared_error')  # Biên dịch lại nếu cần
    return model

model = load_model()

# Function to process image and remove watermark
def remove_watermark(image):
    image = image.resize((196, 196))  # Resize to model's expected input size
    image_array = np.array(image) / 255.0  # Normalize image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    processed_image = model.predict(image_array)
    processed_image = np.squeeze(processed_image, axis=0)  # Remove batch dimension
    print("Output shape:", processed_image.shape)  # Kiểm tra kích thước đầu ra
    processed_image = (processed_image * 255).astype(np.uint8)  # Convert back to uint8 format
    return Image.fromarray(processed_image)
# Image upload section
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Process the image
    if st.button("Remove Watermark"):
        with st.spinner('Removing watermark...'):
            result_image = remove_watermark(image)
            st.image(result_image, caption='Watermark-free Image', use_column_width=True)
        
        # Prepare the image for download as PNG
        buf = io.BytesIO()
        result_image.save(buf, format="PNG")
        byte_im = buf.getvalue()

        # Download option
        st.download_button(
            label="Download image",
            data=byte_im,
            file_name="watermark_free_image.png",
            mime="image/png"
        )
