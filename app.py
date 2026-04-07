import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import xgboost as xgb
import joblib
import os

# Page Config
st.set_page_config(page_title="ML Models", layout="wide")

# Navigation
st.sidebar.title("Projects")
app_mode = st.sidebar.radio("Choose a project:", ["Air Quality Analysis", "Casting Defect Detection"])

# --- PROJECT 1: AIR QUALITY ---
if app_mode == "Air Quality Analysis":
    st.title("Air Quality Prediction")
    st.write("Predicting environmental variables using XGBoost and Random Forest.")
    
    model_type = st.radio("Select Model", ["XGBoost", "Random Forest"])
    
    st.subheader("Enter Input Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        month = st.slider("Month", 1, 12, 6)
        day_of_week = st.slider("Day of Week", 0, 6, 3)
        hour= st.slider("Hour of Day", 0, 23, 12)
        co_gt= st.number_input("CO (GT)", value=2.0)
        nmhc_gt= st.number_input("NMHC (GT)", value=150.0)

    with col2:
        pt08_s1_co= st.number_input("PT08.S1 (CO)", value=1000.0)
        c6h6_gt= st.number_input("C6H6 (GT)", value=8.0)
        pt08_s2_nmhc= st.number_input("PT08.S2 (NMHC)", value=900.0)
        nox_gt= st.number_input("NOx (GT)", value=200.0)

    with col3:
        pt08_s3_nox= st.number_input("PT08.S3 (NOx)", value=700.0)
        no2_gt= st.number_input("NO2 (GT)", value=100.0)
        pt08_s4_no2= st.number_input("PT08.S4 (NO2)", value=1500.0)
        pt08_s5_o3= st.number_input("PT08.S5 (O3)", value=1000.0)

    if st.button("Predict"):
        path = 'xgb_model.json' if model_type == "XGBoost" else 'rf_model.pkl'
        
        if os.path.exists(path):
            if model_type == "XGBoost":
                model = xgb.XGBRegressor()
                model.load_model(path)
            else:
                model = joblib.load(path)
            
            # Create input array (Ensure the shape matches your training data)
            # Placeholder for the rest of your features
            features = np.zeros((1, 13)) 
            features[0, :] = month, day_of_week, hour,co_gt, pt08_s1_co, c6h6_gt, nmhc_gt,  pt08_s2_nmhc,nox_gt, pt08_s3_nox, no2_gt, pt08_s4_no2, pt08_s5_o3
            # ... assign other values ...
            
            prediction = model.predict(features)
            st.success(f"Predicted Value: {prediction[0]}")
        else:
            st.error(f"Model file {path} not found in directory.")

# --- PROJECT 2: CASTING DEFECTS ---
elif app_mode == "Casting Defect Detection":
    st.title("Industrial Casting Defect Detection")
    st.write("Using Computer Vision to identify manufacturing flaws.")

    arch = st.radio("Model Architecture", ["Custom CNN", "MobileNet V2"])
    uploaded_file = st.file_uploader("Upload casting image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Step 1: Read image using OpenCV (No Pillow)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1) # BGR
        
        # Step 2: Display
        st.image(img, channels="BGR", caption="Uploaded Image", width=400)

        # Step 3: Preprocess (matching 224x224 shape, 1/255 scale)
        img_resized = cv2.resize(img, (224, 224))
        img_array = img_resized.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if st.button("Run Diagnostic"):
            model_path = 'custom_cnn.h5' if arch == "Custom CNN" else 'mobilenet_v2.h5'
            
            if os.path.exists(model_path):
                with st.spinner("Analyzing..."):
                    model = tf.keras.models.load_model(model_path)
                    pred = model.predict(img_array)
                    
                    score = pred[0][0]
                    label = "DEFECTIVE" if score > 0.5 else "OK"
                    
                    if label == "DEFECTIVE":
                        st.error(f"Status: {label} (Confidence: {score:.2f})")
                    else:
                        st.success(f"Status: {label} (Confidence: {1-score:.2f})")
            else:
                st.error(f"Model file {model_path} not found.")