import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.preprocessing import load_nifti_file, preprocess_scan, prepare_case_data
from utils.visualization import plot_overlay, plot_probability_maps, SEGMENT_CLASSES
from tensorflow.keras.models import load_model
import tensorflow as tf

# Configuration matching your structure
BRATS_TRAIN_PATH = "BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData"
MODEL_PATH = "Model/model.h5"
IMG_SIZE = 128
VOLUME_START_AT = 22

# Set page config
st.set_page_config(
    page_title="Brain Tumor Segmentation",
    page_icon="🧠",
    layout="wide"
)

@st.experimental_singleton
def load_segmentation_model():
    """Load your trained model with error handling"""
    try:
        model = load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Failed to load model from {MODEL_PATH}: {str(e)}")
        return None

def get_available_cases():
    """List all available cases in your training data"""
    cases = []
    for entry in os.listdir(BRATS_TRAIN_PATH):
        if entry.startswith("BraTS20_Validation_") and os.path.isdir(os.path.join(BRATS_TRAIN_PATH, entry)):
            case_id = entry.replace("BraTS20_Validation_", "")
            cases.append(case_id)
    return sorted(cases)

def main():
    st.title("🧠 Brain Tumor Segmentation")
    st.write("Using your trained model on BraTS2020 data")
    
    # Case selection
    available_cases = get_available_cases()
    if not available_cases:
        st.error("No cases found in the training data directory")
        return
    
    case_id = st.selectbox("Select Case ID", available_cases)
    
    try:
        # Load the case data
        flair_data, t1ce_data = prepare_case_data(BRATS_TRAIN_PATH, case_id)
        
        # Slice selection
        max_slice = flair_data.shape[2] - VOLUME_START_AT - 1
        slice_idx = st.slider("Select Slice", 0, max_slice, min(60, max_slice))
        
        # Display original scans
        st.subheader("Original MRI Scans")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(flair_data[:, :, slice_idx + VOLUME_START_AT], cmap='gray')
        ax1.set_title("FLAIR")
        ax2.imshow(t1ce_data[:, :, slice_idx + VOLUME_START_AT], cmap='gray')
        ax2.set_title("T1CE")
        st.pyplot(fig)
        
        # Segmentation button
        if st.button("Run Segmentation"):
            model = load_segmentation_model()
            if model is None:
                return
            
            with st.spinner("Processing..."):
                # Preprocess the selected slice
                scan = preprocess_scan(flair_data, t1ce_data, slice_idx + VOLUME_START_AT)
                input_data = np.expand_dims(scan, axis=0)
                
                # Predict
                prediction = model.predict(input_data)
                predicted_mask = np.argmax(prediction[0], axis=-1)
                
                # Display results
                st.subheader("Segmentation Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### Tumor Segmentation Overlay")
                    fig = plot_overlay(scan, predicted_mask)
                    st.pyplot(fig)
                
                with col2:
                    st.write("### Probability Maps")
                    fig = plot_probability_maps(prediction)
                    st.pyplot(fig)
                
                # Volume calculations
                voxel_size = 1.0  # mm³, adjust if you know your actual voxel size
                volumes = {
                    class_name: np.sum(predicted_mask == class_idx) * voxel_size
                    for class_idx, class_name in SEGMENT_CLASSES.items()
                    if class_idx != 0  # Skip background
                }
                
                st.subheader("Tumor Volume Estimates")
                cols = st.columns(3)
                for (class_name, volume), col in zip(volumes.items(), cols):
                    col.metric(f"{class_name} Volume", f"{volume:.2f} mm³")
    
    except Exception as e:
        st.error(f"Error processing case {case_id}: {str(e)}")

if __name__ == "__main__":
    main()