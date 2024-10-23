import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub  # Thêm tensorflow_hub để tải mô hình Super Resolution
import io
import cv2

# Load watermark removal model
@st.cache_resource
def load_watermark_removal_model():
    model = tf.keras.models.load_model('watermark_auto_encoder_model.h5', compile=False)
    model.compile(optimizer='adam', loss='mean_squared_error')  # Biên dịch lại nếu cần
    return model

# Load Super Resolution model (ESRGAN/EDSR)
@st.cache_resource
def load_super_resolution_model():
    # Load mô hình ESRGAN từ TensorFlow Hub
    sr_model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")
    return sr_model

# Function to apply super resolution
def apply_super_resolution(image, sr_model):
    """
    Áp dụng mô hình Super Resolution để tăng cường độ phân giải của ảnh.
    """
    img = tf.convert_to_tensor(image, dtype=tf.float32)
    img = tf.image.resize(img, [tf.shape(img)[0], tf.shape(img)[1]])  # Resize lại ảnh nếu cần
    img = tf.expand_dims(img, axis=0)  # Thêm batch dimension
    sr_img = sr_model(img)  # Áp dụng Super Resolution
    sr_img = tf.squeeze(sr_img)  # Bỏ batch dimension
    return sr_img.numpy().astype(np.uint8)  # Trả về ảnh dạng numpy array

# Function to process image and remove watermark
# Thay đổi kích thước ảnh đầu vào
def remove_watermark(image, model):
    image = image.resize((196, 196))  # Resize lên kích thước lớn hơn
    image_array = np.array(image) / 255.0  # Chuẩn hóa ảnh
    image_array = np.expand_dims(image_array, axis=0)  # Thêm batch dimension
    processed_image = model.predict(image_array)
    processed_image = np.squeeze(processed_image, axis=0)  # Bỏ batch dimension
    processed_image = (processed_image * 255).astype(np.uint8)  # Chuyển ảnh về uint8
    return Image.fromarray(processed_image)
# Hàm làm sắc nét ảnh trước khi áp dụng Super Resolution
def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])  # Kernel làm sắc nét ảnh
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

# Tích hợp vào pipeline
def process_image(image, model, sr_model):
    # Xóa watermark
    result_image = remove_watermark(image, model)
    
    # Làm sắc nét ảnh trước khi áp dụng Super Resolution
    result_image_np = np.array(result_image)  # Chuyển đổi PIL image sang numpy array
    sharpened_image = sharpen_image(result_image_np)
    
    # Áp dụng Super Resolution
    sr_result_image = apply_super_resolution(sharpened_image, sr_model)
    
    return Image.fromarray(sr_result_image)

# Load models
watermark_removal_model = load_watermark_removal_model()
super_resolution_model = load_super_resolution_model()

# Image upload section
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Process the image
    if st.button("Remove Watermark and Enhance Resolution"):
        with st.spinner('Processing...'):
            # Step 1: Remove the watermark
            result_image = remove_watermark(image, watermark_removal_model)

            # Step 2: Apply Super Resolution
            result_image_np = np.array(result_image)  # Chuyển đổi PIL image sang numpy array
            sr_result_image = apply_super_resolution(result_image_np, super_resolution_model)

            # Convert numpy array back to PIL image for displaying
            sr_result_image_pil = Image.fromarray(sr_result_image)

            # Display the final result (watermark removed and resolution enhanced)
            st.image(sr_result_image_pil, caption='Watermark-free and Enhanced Image', use_column_width=True)

        # Prepare the image for download as PNG
        buf = io.BytesIO()
        sr_result_image_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()

        # Download option
        st.download_button(
            label="Download image",
            data=byte_im,
            file_name="watermark_free_and_enhanced_image.png",
            mime="image/png"
        )