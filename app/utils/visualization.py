import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Tumor class definitions matching your training
SEGMENT_CLASSES = {
    0: 'NOT tumor',
    1: 'NECROTIC/CORE',
    2: 'EDEMA',
    3: 'ENHANCING'
}

COLOR_MAP = mcolors.ListedColormap([
    [0, 0, 0, 0],      # Transparent for background
    [1, 0, 0, 0.6],    # Red for necrotic/core
    [0, 1, 0, 0.6],    # Green for edema
    [0, 0, 1, 0.6]     # Blue for enhancing
])

def plot_overlay(scan_slice, mask_slice):
    """Create overlay plot of scan and segmentation mask"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Display the FLAIR channel
    ax.imshow(scan_slice[:, :, 0], cmap='gray')
    
    # Create colored mask overlay
    mask_rgba = COLOR_MAP(mask_slice)
    ax.imshow(mask_rgba)
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_MAP.colors[i], label=SEGMENT_CLASSES[i])
        for i in range(1, 4)  # Skip background
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def plot_probability_maps(prediction, slice_idx=0):
    """Plot probability maps for each class"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ['No Tumor', 'Necrotic/Core', 'Edema', 'Enhancing']
    
    for i, ax in enumerate(axes):
        prob_map = prediction[slice_idx, :, :, i]
        im = ax.imshow(prob_map, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(titles[i])
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig