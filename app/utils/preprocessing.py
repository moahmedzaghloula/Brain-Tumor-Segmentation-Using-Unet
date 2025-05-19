import os
import cv2
import numpy as np
import nibabel as nib

# Configuration matching your Validation
IMG_SIZE = 128
VOLUME_SLICES = 100
VOLUME_START_AT = 22

def load_nifti_file(file_path):
    """Load NIfTI file and return its data with error handling"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"NIfTI file not found: {file_path}")
    try:
        img = nib.load(file_path)
        return img.get_fdata()
    except Exception as e:
        raise ValueError(f"Error loading NIfTI file {file_path}: {str(e)}")

def preprocess_scan(flair_data, t1ce_data, slice_idx):
    """Preprocess a single slice exactly as in your Validation"""
    if slice_idx >= flair_data.shape[2]:
        raise ValueError(f"Slice index {slice_idx} exceeds volume depth {flair_data.shape[2]}")
    
    # Resize to 128x128 as in your model
    flair_slice = cv2.resize(flair_data[:, :, slice_idx], (IMG_SIZE, IMG_SIZE))
    t1ce_slice = cv2.resize(t1ce_data[:, :, slice_idx], (IMG_SIZE, IMG_SIZE))
    
    # Normalization matching your training
    flair_slice = flair_slice / (np.max(flair_slice) + 1e-8)
    t1ce_slice = t1ce_slice / (np.max(t1ce_slice) + 1e-8)
    
    return np.stack([flair_slice, t1ce_slice], axis=-1)

def prepare_case_data(base_path, case_id):
    """Load and validate all data for a case"""
    case_path = os.path.join(base_path, f"BraTS20_Validation_{case_id}")
    if not os.path.exists(case_path):
        raise FileNotFoundError(f"Case directory not found: {case_path}")
    
    flair_path = os.path.join(case_path, f"BraTS20_Validation_{case_id}_flair.nii")
    t1ce_path = os.path.join(case_path, f"BraTS20_Validation_{case_id}_t1ce.nii")
    
    flair_data = load_nifti_file(flair_path)
    t1ce_data = load_nifti_file(t1ce_path)
    
    return flair_data, t1ce_data