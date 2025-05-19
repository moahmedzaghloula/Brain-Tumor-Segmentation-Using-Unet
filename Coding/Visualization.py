import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from skimage.transform import rotate
from skimage.util import montage
import nibabel as nib
import nilearn as nl
import nilearn.plotting as nlplt
import SimpleITK as sitk

# Define constants
SEGMENT_CLASSES = {
    0: 'Not Tumor',
    1: 'Necrotic/Core',
    2: 'Edema',
    3: 'Enhancing'
}
VOLUME_SLICES = 100
SLICE_OFFSET = 25
DATA_ROOT = os.path.expanduser('~/brain_tumor_segmentation_update')
TRAIN_DATA_PATH = os.path.join(DATA_ROOT, 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData')
VISUALIZATION_DIR = os.path.join(DATA_ROOT, 'Visualizations')

# Ensure visualization directory exists
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def load_nifti_image(file_path):
    """Load a NIfTI image and return its data as a numpy array."""
    return nib.load(file_path).get_fdata()

def load_nilearn_image(file_path):
    """Load a NIfTI image using Nilearn."""
    return nl.image.load_img(file_path)

def load_sitk_image(file_path):
    """Load an image using SimpleITK and return its data as a numpy array."""
    return sitk.GetArrayFromImage(sitk.ReadImage(file_path))

def save_plot(fig, filename):
    """Save the figure to the visualizations directory."""
    fig.savefig(os.path.join(VISUALIZATION_DIR, filename), bbox_inches='tight', dpi=300)
    plt.close(fig)

# Visualization 1: Display different MRI modalities with segmentation mask
def plot_mri_modalities():
    subject_id = 'BraTS20_Training_001'
    modalities = {
        'flair': f'{subject_id}_flair.nii',
        't1': f'{subject_id}_t1.nii',
        't1ce': f'{subject_id}_t1ce.nii',
        't2': f'{subject_id}_t2.nii',
        'seg': f'{subject_id}_seg.nii'
    }
    
    # Load images
    images = {mod: load_nifti_image(os.path.join(TRAIN_DATA_PATH, subject_id, file)) 
              for mod, file in modalities.items()}
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    slice_idx = images['flair'].shape[0] // 2 - SLICE_OFFSET
    
    for ax, (mod, img) in zip(axes, images.items()):
        if mod == 'seg':
            ax.imshow(img[:, :, slice_idx], cmap='jet')
        else:
            ax.imshow(img[:, :, slice_idx], cmap='gray')
        ax.set_title(mod.upper())
        ax.axis('off')
    
    plt.suptitle(f'MRI Modalities and Segmentation - {subject_id}')
    save_plot(fig, 'mri_modalities.png')

# Visualization 2: Three-view T1CE display
def plot_three_views():
    t1ce_path = os.path.join(TRAIN_DATA_PATH, 'BraTS20_Training_001/BraTS20_Training_001_t1ce.nii')
    t1ce_img = load_nifti_image(t1ce_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Transverse
    axes[0].imshow(t1ce_img[:, :, VOLUME_SLICES], cmap='gray')
    axes[0].set_title('Transverse View')
    
    # Frontal
    axes[1].imshow(rotate(t1ce_img[:, VOLUME_SLICES, :], 90, resize=True), cmap='gray')
    axes[1].set_title('Frontal View')
    
    # Sagittal
    axes[2].imshow(rotate(t1ce_img[VOLUME_SLICES, :, :], 90, resize=True), cmap='gray')
    axes[2].set_title('Sagittal View')
    
    plt.suptitle('T1CE Three Views')
    save_plot(fig, 'three_views.png')

# Visualization 3: Segmentation mask with custom colormap
def plot_segmentation_mask():
    mask_path = os.path.join(TRAIN_DATA_PATH, 'BraTS20_Training_001/BraTS20_Training_001_seg.nii')
    mask = load_nifti_image(mask_path)
    
    cmap = mcolors.ListedColormap(['#440054', '#3b528b', '#18b880', '#e6d74f'])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mask[:, :, 50], cmap=cmap, norm=norm)
    plt.colorbar(ax.imshow(mask[:, :, 50], cmap=cmap, norm=norm))
    ax.set_title('Segmentation Mask - Slice 50')
    ax.axis('off')
    
    save_plot(fig, 'segmentation_mask.png')

# Visualization 4: MRI and mask overlay
def plot_mri_mask_overlay():
    subject_id = 'BraTS20_Training_011'
    flair_path = os.path.join(TRAIN_DATA_PATH, subject_id, f'{subject_id}_flair.nii')
    mask_path = os.path.join(TRAIN_DATA_PATH, subject_id, f'{subject_id}_seg.nii')
    
    flair_slice = load_sitk_image(flair_path)[50, :, :]
    mask_slice = load_sitk_image(mask_path)[50, :, :]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(flair_slice, cmap='gray')
    axes[0].set_title('FLAIR MRI')
    axes[1].imshow(flair_slice, cmap='gray')
    axes[1].imshow(mask_slice, cmap='jet', alpha=0.5)
    axes[1].set_title('Segmentation Overlay')
    
    plt.suptitle(f'MRI and Segmentation - {subject_id}')
    save_plot(fig, 'mri_mask_overlay.png')

# Visualization 5: Multi-axis slices
def plot_multi_axis_slices():
    subject_id = 'BraTS20_Training_008'
    flair_path = os.path.join(TRAIN_DATA_PATH, subject_id, f'{subject_id}_flair.nii')
    mask_path = os.path.join(TRAIN_DATA_PATH, subject_id, f'{subject_id}_seg.nii')
    
    flair_img = load_nifti_image(flair_path)
    mask_img = load_nifti_image(mask_path)
    
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    
    # X-axis
    ax[0, 0].imshow(flair_img[VOLUME_SLICES, :, :], cmap='gray')
    ax[0, 0].set_title(f'FLAIR Slice {VOLUME_SLICES} (X-axis)')
    ax[1, 0].imshow(mask_img[VOLUME_SLICES, :, :], cmap='jet')
    ax[1, 0].set_title(f'Mask Slice {VOLUME_SLICES} (X-axis)')
    
    # Y-axis
    ax[0, 1].imshow(flair_img[:, VOLUME_SLICES, :], cmap='gray')
    ax[0, 1].set_title(f'FLAIR Slice {VOLUME_SLICES} (Y-axis)')
    ax[1, 1].imshow(mask_img[:, VOLUME_SLICES, :], cmap='jet')
    ax[1, 1].set_title(f'Mask Slice {VOLUME_SLICES} (Y-axis)')
    
    # Z-axis
    ax[0, 2].imshow(flair_img[:, :, VOLUME_SLICES], cmap='gray')
    ax[0, 2].set_title(f'FLAIR Slice {VOLUME_SLICES} (Z-axis)')
    ax[1, 2].imshow(mask_img[:, :, VOLUME_SLICES], cmap='jet')
    ax[1, 2].set_title(f'Mask Slice {VOLUME_SLICES} (Z-axis)')
    
    plt.suptitle(f'Multi-axis Slices - {subject_id}')
    save_plot(fig, 'multi_axis_slices.png')

# Visualization 6: Multimodal scans with tumor regions
def plot_multimodal_tumor():
    subject_id = 'BraTS20_Training_001'
    modalities = {
        'flair': load_nifti_image(os.path.join(TRAIN_DATA_PATH, subject_id, f'{subject_id}_flair.nii')),
        't1': load_nifti_image(os.path.join(TRAIN_DATA_PATH, subject_id, f'{subject_id}_t1.nii')),
        't2': load_nifti_image(os.path.join(TRAIN_DATA_PATH, subject_id, f'{subject_id}_t2.nii')),
        't1ce': load_nifti_image(os.path.join(TRAIN_DATA_PATH, subject_id, f'{subject_id}_t1ce.nii')),
        'seg': load_nifti_image(os.path.join(TRAIN_DATA_PATH, subject_id, f'{subject_id}_seg.nii'))
    }
    
    # Create tumor masks
    mask_wt = modalities['seg'].copy()
    mask_wt[mask_wt == 1] = 1
    mask_wt[mask_wt == 2] = 1
    mask_wt[mask_wt == 4] = 1
    
    mask_tc = modalities['seg'].copy()
    mask_tc[mask_tc == 1] = 1
    mask_tc[mask_tc == 2] = 0
    mask_tc[mask_tc == 4] = 1
    
    mask_et = modalities['seg'].copy()
    mask_et[mask_et == 1] = 0
    mask_et[mask_et == 2] = 0
    mask_et[mask_et == 4] = 1
    
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(nrows=2, ncols=4, height_ratios=[1, 1.5])
    
    # Plot modalities
    for i, mod in enumerate(['flair', 't1', 't2', 't1ce']):
        ax = fig.add_subplot(gs[0, i])
        img = ax.imshow(modalities[mod][:, :, 65], cmap='bone')
        ax.set_title(mod.upper(), fontsize=18, weight='bold', y=-0.2)
        fig.colorbar(img)
        ax.axis('off')
    
    # Plot tumor regions
    ax = fig.add_subplot(gs[1, 1:3])
    l1 = ax.imshow(mask_wt[:, :, 65], cmap='summer')
    l2 = ax.imshow(np.ma.masked_where(mask_tc[:, :, 65] == False, mask_tc[:, :, 65]), cmap='rainbow', alpha=0.6)
    l3 = ax.imshow(np.ma.masked_where(mask_et[:, :, 65] == False, mask_et[:, :, 65]), cmap='winter', alpha=0.6)
    ax.axis('off')
    
    # Add legend
    colors = [im.cmap(im.norm(1)) for im in [l1, l2, l3]]
    labels = ['Whole Tumor', 'Tumor Core', 'Enhancing Tumor']
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
    plt.legend(handles=patches, bbox_to_anchor=(1.1, 0.65), loc=2, borderaxespad=0.4, 
              fontsize='xx-large', title='Tumor Regions', title_fontsize=18, 
              edgecolor="black", facecolor='#c5c6c7')
    
    plt.suptitle(f'Multimodal Scans with Tumor Regions - {subject_id}', fontsize=20, weight='bold')
    save_plot(fig, 'multimodal_tumor.png')

# Visualization 7: T1 Montage and Nilearn Plots
def plot_montage_and_nilearn():
    subject_id = 'BraTS20_Training_001'
    t1_path = os.path.join(TRAIN_DATA_PATH, subject_id, f'{subject_id}_t1.nii')
    flair_path = os.path.join(TRAIN_DATA_PATH, subject_id, f'{subject_id}_flair.nii')
    seg_path = os.path.join(TRAIN_DATA_PATH, subject_id, f'{subject_id}_seg.nii')
    
    # Montage plot
    t1_img = load_nifti_image(t1_path)
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.imshow(rotate(montage(t1_img[50:-50, :, :]), 90, resize=True), cmap='gray')
    ax.set_title(f'T1 Montage - {subject_id}')
    ax.axis('off')
    save_plot(fig, 't1_montage.png')
    
    # Nilearn plots
    niimg = load_nilearn_image(flair_path)
    nimask = load_nilearn_image(seg_path)
    
    fig, axes = plt.subplots(nrows=4, figsize=(30, 40))
    
    nlplt.plot_anat(niimg, title=f'{subject_id}_flair.nii - Anatomical Plot', axes=axes[0])
    nlplt.plot_epi(niimg, title=f'{subject_id}_flair.nii - EPI Plot', axes=axes[1])
    nlplt.plot_img(niimg, title=f'{subject_id}_flair.nii - Image Plot', axes=axes[2])
    nlplt.plot_roi(nimask, title=f'{subject_id}_flair.nii with Segmentation - ROI Plot', 
                  bg_img=niimg, axes=axes[3], cmap='Paired')
    
    save_plot(fig, 'nilearn_plots.png')

# Visualization 8: Four Modalities Display
def plot_four_modalities():
    subject_id = 'BraTS20_Training_020'
    modalities = {
        'FLAIR': f'{subject_id}_flair.nii',
        'T1': f'{subject_id}_t1.nii',
        'T1CE': f'{subject_id}_t1ce.nii',
        'T2': f'{subject_id}_t2.nii'
    }
    
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    for ax, (title, filename) in zip(axes, modalities.items()):
        path = os.path.join(TRAIN_DATA_PATH, subject_id, filename)
        image_array = load_sitk_image(path)
        ax.imshow(image_array[VOLUME_SLICES, :, :], cmap='nipy_spectral')
        ax.set_title(title)
        ax.axis('off')
    
    plt.suptitle(f'Four Modalities - {subject_id}')
    plt.tight_layout()
    save_plot(fig, 'four_modalities.png')

# Main execution
if __name__ == '__main__':
    plot_mri_modalities()
    plot_three_views()
    plot_segmentation_mask()
    plot_mri_mask_overlay()
    plot_multi_axis_slices()
    plot_multimodal_tumor()
    plot_montage_and_nilearn()
    plot_four_modalities()
    print(f"All visualizations saved to {VISUALIZATION_DIR}")
