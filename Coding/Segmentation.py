# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import random
import cv2
import nibabel as nib
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras import backend as K
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate
from tensorflow.keras.utils import plot_model
import matplotlib.colors as mcolors
import glob
import re

# Test GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Found {len(gpus)} GPU(s):")
    for i, gpu in enumerate(gpus):
        print(f"GPU {i}: {gpu.name}")
    strategy = tf.distribute.MirroredStrategy()
else:
    print("No GPUs found. Running on CPU.")
    strategy = tf.distribute.get_strategy()

# Define constants
DATA_ROOT = "/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData"
TRAIN_DATASET_PATH = os.path.join(DATA_ROOT, "MICCAI_BraTS2020_TrainingData")
VISUALIZATION_DIR = "/kaggle/working/Visualizations"
PREDICTION_DIR = "/kaggle/working/Predictions"
CODING_DIR = "/kaggle/working/Coding"
IMG_SIZE = 128
VOLUME_SLICES = 100
VOLUME_START_AT = 22
BASE_BATCH_SIZE = 2
NUM_GPUS = len(gpus) if gpus else 1
BATCH_SIZE = BASE_BATCH_SIZE * NUM_GPUS
SEGMENT_CLASSES = {
    0: 'NOT tumor',
    1: 'NECROTIC/CORE',
    2: 'EDEMA',
    3: 'ENHANCING'
}

def get_directory_ids(dir_path):
    """Retrieve list of directory names from the given path."""
    try:
        directories = [f.path for f in os.scandir(dir_path) if f.is_dir()]
        if not directories:
            raise ValueError(f"No directories found in {dir_path}")
        return directories
    except FileNotFoundError:
        raise FileNotFoundError(f"Directory not found: {dir_path}. Ensure the BraTS2020 dataset is correctly placed.")
    except Exception as e:
        raise Exception(f"Error accessing {dir_path}: {str(e)}")

def path_list_to_ids(dir_list):
    """Convert directory paths to IDs by extracting the last component."""
    return [os.path.basename(path).replace('BraTS20_Training_', '').replace('BraTS20_Validation_', '') for path in dir_list]

def split_dataset():
    """Split dataset into train, validation, and test sets."""
    try:
        train_and_val_directories = get_directory_ids(TRAIN_DATASET_PATH)
    except Exception as e:
        print(f"Error: {str(e)}")
        return None, None, None
    
    train_and_test_ids = path_list_to_ids(train_and_val_directories)
    print(f"Found {len(train_and_test_ids)} subjects in the dataset.")
    
    train_test_ids, val_ids = train_test_split(train_and_test_ids, test_size=0.2, random_state=42)
    train_ids, test_ids = train_test_split(train_test_ids, test_size=0.15, random_state=42)
    
    print(f"Train set: {len(train_ids)} subjects")
    print(f"Validation set: {len(val_ids)} subjects")
    print(f"Test set: {len(test_ids)} subjects")
    
    return train_ids, val_ids, test_ids

def plot_data_distribution(train_ids, val_ids, test_ids):
    """Create and save a bar plot of the dataset distribution."""
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    plt.bar(
        ["Train", "Valid", "Test"],
        [len(train_ids), len(val_ids), len(test_ids)],
        align='center',
        color=['green', 'red', 'blue'],
        label=["Train", "Valid", "Test"]
    )
    
    plt.legend()
    plt.ylabel('Number of Images')
    plt.title('Data Distribution')
    
    plot_path = os.path.join(VISUALIZATION_DIR, "data_distribution.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved data distribution plot to {plot_path}")

class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras."""
    def __init__(self, list_IDs, dim=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, n_channels=2, shuffle=True):
        """Initialization."""
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_ids = [self.list_IDs[k] for k in indexes]
        X, Y = self.__data_generation(batch_ids)
        return X, Y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_ids):
        """Generates data containing batch_size samples."""
        X = np.zeros((self.batch_size * VOLUME_SLICES, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size * VOLUME_SLICES, 240, 240))
        Y = np.zeros((self.batch_size * VOLUME_SLICES, *self.dim, 4))

        for c, subject_id in enumerate(batch_ids):
            case_path = os.path.join(TRAIN_DATASET_PATH, f'BraTS20_Training_{subject_id}')
            
            try:
                flair_path = os.path.join(case_path, f'BraTS20_Training_{subject_id}_flair.nii')
                if not os.path.exists(flair_path):
                    raise FileNotFoundError(f"FLAIR file missing: {flair_path}")
                flair = nib.load(flair_path).get_fdata()

                t1ce_path = os.path.join(case_path, f'BraTS20_Training_{subject_id}_t1ce.nii')
                if not os.path.exists(t1ce_path):
                    raise FileNotFoundError(f"T1CE file missing: {t1ce_path}")
                t1ce = nib.load(t1ce_path).get_fdata()

                seg_path = os.path.join(case_path, f'BraTS20_Training_{subject_id}_seg.nii')
                if not os.path.exists(seg_path):
                    raise FileNotFoundError(f"Segmentation file missing: {seg_path}")
                seg = nib.load(seg_path).get_fdata()

                for j in range(VOLUME_SLICES):
                    slice_idx = j + VOLUME_START_AT
                    if slice_idx >= flair.shape[2]:
                        continue
                    
                    flair_slice = cv2.resize(flair[:, :, slice_idx], self.dim)
                    t1ce_slice = cv2.resize(t1ce[:, :, slice_idx], self.dim)
                    X[j + VOLUME_SLICES * c, :, :, 0] = flair_slice / np.max(flair_slice) if np.max(flair_slice) != 0 else flair_slice
                    X[j + VOLUME_SLICES * c, :, :, 1] = t1ce_slice / np.max(t1ce_slice) if np.max(t1ce_slice) != 0 else t1ce_slice

                    y[j + VOLUME_SLICES * c] = seg[:, :, slice_idx]

            except FileNotFoundError as e:
                print(f"Error: {str(e)}")
                continue

        y[y == 4] = 3
        mask = tf.one_hot(y, 4)
        Y = tf.image.resize(mask, self.dim)

        return X, Y

def visualize_sample_batch(generator, num_samples=3):
    """Visualize a few samples from the data generator."""
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    
    X, Y = generator[0]
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    for i in range(min(num_samples, len(X))):
        axes[i, 0].imshow(X[i, :, :, 0], cmap='gray')
        axes[i, 0].set_title('FLAIR')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(X[i, :, :, 1], cmap='gray')
        axes[i, 1].set_title('T1CE')
        axes[i, 1].axis('off')
        
        seg_mask = np.argmax(Y[i], axis=-1)
        axes[i, 2].imshow(seg_mask, cmap='jet', vmin=0, vmax=3)
        axes[i, 2].set_title('Segmentation Mask')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(VISUALIZATION_DIR, "sample_batch.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved sample batch visualization to {plot_path}")

def dice_coef(y_true, y_pred, smooth=1e-6):
    import tensorflow as tf
    class_num = 4
    dice = 0
    for i in range(class_num):
        y_true_f = tf.reshape(y_true[:, :, :, i], [-1])
        y_pred_f = tf.reshape(y_pred[:, :, :, i], [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
        dice += (2. * intersection + smooth) / (union + smooth)
    return dice / class_num

def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    import tensorflow as tf
    intersection = tf.reduce_sum(tf.abs(y_true[:, :, :, 1] * y_pred[:, :, :, 1]))
    return (2. * intersection) / (tf.reduce_sum(tf.square(y_true[:, :, :, 1])) + tf.reduce_sum(tf.square(y_pred[:, :, :, 1])) + epsilon)
def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    import tensorflow as tf
    intersection = tf.reduce_sum(tf.abs(y_true[:, :, :, 2] * y_pred[:, :, :, 2]))
    return (2. * intersection) / (tf.reduce_sum(tf.square(y_true[:, :, :, 2])) + tf.reduce_sum(tf.square(y_pred[:, :, :, 2])) + epsilon)

def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    intersection = tf.reduce_sum(tf.abs(y_true[:,:,:,3] * y_pred[:,:,:,3]))
    return (2. * intersection) / (tf.reduce_sum(tf.square(y_true[:,:,:,3])) + tf.reduce_sum(tf.square(y_pred[:,:,:,3])) + epsilon)

def precision(y_true, y_pred):
    import tensorflow as tf
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    return true_positives / (predicted_positives + tf.keras.backend.epsilon())

def sensitivity(y_true, y_pred):
    import tensorflow as tf
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    return true_positives / (possible_positives + tf.keras.backend.epsilon())

def specificity(y_true, y_pred):
    import tensorflow as tf
    true_negatives = tf.reduce_sum(tf.round(tf.clip_by_value((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = tf.reduce_sum(tf.round(tf.clip_by_value(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + tf.keras.backend.epsilon())

def build_unet(inputs, ker_init, dropout):
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=ker_init)(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv1)
    pool = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=ker_init)(pool)
    conv = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=ker_init)(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=ker_init)(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=ker_init)(pool4)
    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv5)
    drop5 = Dropout(dropout)(conv5)
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=ker_init)(UpSampling2D(size=(2,2))(drop5))
    merge7 = concatenate([conv3,up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=ker_init)(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv7)
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=ker_init)(UpSampling2D(size=(2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=ker_init)(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv8)
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=ker_init)(UpSampling2D(size=(2,2))(conv8))
    merge9 = concatenate([conv,up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=ker_init)(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv9)
    up = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer=ker_init)(UpSampling2D(size=(2,2))(conv9))
    merge = concatenate([conv1,up], axis=3)
    conv = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=ker_init)(merge)
    conv = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv)
    conv10 = Conv2D(4, (1,1), activation='softmax')(conv)
    return Model(inputs=inputs, outputs=conv10)

def imageLoader(path):
    """Load a NIfTI file and return its data."""
    return nib.load(path).get_fdata()

def loadDataFromDir(path_list, mriType, n_images):
    """Load scans and masks from directories."""
    scans = []
    masks = []
    for i in path_list[:n_images]:
        case_path = os.path.join(TRAIN_DATASET_PATH, f'BraTS20_Training_{i}')
        flair_path = os.path.join(case_path, f'BraTS20_Training_{i}_flair.nii')
        t1ce_path = os.path.join(case_path, f'BraTS20_Training_{i}_t1ce.nii')
        seg_path = os.path.join(case_path, f'BraTS20_Training_{i}_seg.nii')
        
        try:
            flair = imageLoader(flair_path)
            t1ce = imageLoader(t1ce_path)
            seg = imageLoader(seg_path)
            
            for j in range(VOLUME_SLICES):
                slice_idx = j + VOLUME_START_AT
                if slice_idx >= flair.shape[2]:
                    continue
                scan_img = cv2.resize(flair[:,:,slice_idx], dsize=(IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_AREA)
                t1ce_img = cv2.resize(t1ce[:,:,slice_idx], dsize=(IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_AREA)
                mask_img = cv2.resize(seg[:,:,slice_idx], dsize=(IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_NEAREST)
                scans.append(np.stack([scan_img, t1ce_img], axis=-1))
                masks.append(mask_img)
        except FileNotFoundError as e:
            print(f"Error: {str(e)}")
            continue
    
    return np.array(scans, dtype='float32') / np.max(scans), np.array(masks, dtype='float32')

def predictByPath(model, case_path, case):
    """Predict segmentation for a case."""
    X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2))
    
    try:
        flair_path = os.path.join(case_path, f'BraTS20_Training_{case}_flair.nii')
        flair = nib.load(flair_path).get_fdata()
        
        t1ce_path = os.path.join(case_path, f'BraTS20_Training_{case}_t1ce.nii')
        t1ce = nib.load(t1ce_path).get_fdata()
        
        for j in range(VOLUME_SLICES):
            slice_idx = j + VOLUME_START_AT
            if slice_idx >= flair.shape[2]:
                continue
            X[j,:,:,0] = cv2.resize(flair[:,:,slice_idx], (IMG_SIZE,IMG_SIZE))
            X[j,:,:,1] = cv2.resize(t1ce[:,:,slice_idx], (IMG_SIZE,IMG_SIZE))
        
        return model.predict(X/np.max(X), verbose=1, batch_size=BATCH_SIZE)
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        return None

def showPredictsById(model, case, start_slice=60):
    """Visualize predictions for a case."""
    path = os.path.join(TRAIN_DATASET_PATH, f'BraTS20_Training_{case}')
    try:
        gt = nib.load(os.path.join(path, f'BraTS20_Training_{case}_seg.nii')).get_fdata()
        origImage = nib.load(os.path.join(path, f'BraTS20_Training_{case}_flair.nii')).get_fdata()
        p = predictByPath(model, path, case)
        
        if p is None:
            print(f"Skipping visualization for case {case} due to prediction error.")
            return
        
        core = p[:,:,:,1]
        edema = p[:,:,:,2]
        enhancing = p[:,:,:,3]
        
        plt.figure(figsize=(18, 5))
        f, axarr = plt.subplots(1, 6, figsize=(18, 5))
        
        for i in range(6):
            axarr[i].imshow(cv2.resize(origImage[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray", interpolation='none')
        
        axarr[0].imshow(cv2.resize(origImage[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray")
        axarr[0].title.set_text('Original image flair')
        curr_gt = cv2.resize(gt[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        axarr[1].imshow(curr_gt, cmap="Reds", interpolation='none', alpha=0.3)
        axarr[1].title.set_text('Ground truth')
        axarr[2].imshow(p[start_slice,:,:,1:4], cmap="Reds", interpolation='none', alpha=0.3)
        axarr[2].title.set_text('All classes predicted')
        axarr[3].imshow(edema[start_slice,:,:], cmap="OrRd", interpolation='none', alpha=0.3)
        axarr[3].title.set_text(f'{SEGMENT_CLASSES[2]} predicted')
        axarr[4].imshow(core[start_slice,:,:], cmap="OrRd", interpolation='none', alpha=0.3)
        axarr[4].title.set_text(f'{SEGMENT_CLASSES[1]} predicted')
        axarr[5].imshow(enhancing[start_slice,:,:], cmap="OrRd", interpolation='none', alpha=0.3)
        axarr[5].title.set_text(f'{SEGMENT_CLASSES[3]} predicted')
        
        os.makedirs(PREDICTION_DIR, exist_ok=True)
        plot_path = os.path.join(PREDICTION_DIR, f'prediction_case_{case}_slice_{start_slice}.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved prediction visualization for case {case} to {plot_path}")
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")

def predict_segmentation(model, sample_path):
    """Predict segmentation for a sample."""
    t1ce_path = sample_path + '_t1ce.nii'
    flair_path = sample_path + '_flair.nii'
    
    try:
        t1ce = nib.load(t1ce_path).get_fdata()
        flair = nib.load(flair_path).get_fdata()
        
        X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2))
        for j in range(VOLUME_SLICES):
            slice_idx = j + VOLUME_START_AT
            if slice_idx >= flair.shape[2]:
                continue
            X[j,:,:,0] = cv2.resize(flair[:,:,slice_idx], (IMG_SIZE,IMG_SIZE))
            X[j,:,:,1] = cv2.resize(t1ce[:,:,slice_idx], (IMG_SIZE,IMG_SIZE))
        
        return model.predict(X/np.max(X), verbose=1, batch_size=BATCH_SIZE)
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        return None

def show_predicted_segmentations(model, samples_list, slice_to_plot, cmap, norm):
    """Visualize predicted segmentations for a random sample."""
    random_sample = random.choice(samples_list)
    random_sample_path = os.path.join(TRAIN_DATASET_PATH, f'BraTS20_Training_{random_sample}', f'BraTS20_Training_{random_sample}')
    
    predicted_seg = predict_segmentation(model, random_sample_path)
    if predicted_seg is None:
        print(f"Skipping visualization for sample {random_sample} due to prediction error.")
        return
    
    seg_path = random_sample_path + '_seg.nii'
    try:
        seg = nib.load(seg_path).get_fdata()
        seg = cv2.resize(seg[:,:,slice_to_plot+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        
        all_classes = predicted_seg[slice_to_plot,:,:,1:4]
        zero = predicted_seg[slice_to_plot,:,:,0]
        first = predicted_seg[slice_to_plot,:,:,1]
        second = predicted_seg[slice_to_plot,:,:,2]
        third = predicted_seg[slice_to_plot,:,:,3]
        
        fig, axstest = plt.subplots(1, 6, figsize=(25, 5))
        axstest[0].imshow(seg, cmap=cmap, norm=norm)
        axstest[0].set_title('Original Segmentation')
        axstest[1].imshow(all_classes, cmap=cmap, norm=norm)
        axstest[1].set_title('Predicted Segmentation - all classes')
        axstest[2].imshow(zero, cmap='gray')
        axstest[2].set_title('Predicted Segmentation - Not Tumor')
        axstest[3].imshow(first, cmap='gray')
        axstest[3].set_title('Predicted Segmentation - Necrotic/Core')
        axstest[4].imshow(second, cmap='gray')
        axstest[4].set_title('Predicted Segmentation - Edema')
        axstest[5].imshow(third, cmap='gray')
        axstest[5].set_title('Predicted Segmentation - Enhancing')
        
        plt.subplots_adjust(wspace=0.8)
        os.makedirs(PREDICTION_DIR, exist_ok=True)
        plot_path = os.path.join(PREDICTION_DIR, f'predicted_seg_{random_sample}_slice_{slice_to_plot}.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Patient number: {random_sample}")
        print(f"Saved predicted segmentation visualization to {plot_path}")
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")

def evaluate_single_class(model, case, eval_class, slice_idx=40):
    """Evaluate a single class for a case."""
    path = os.path.join(TRAIN_DATASET_PATH, f'BraTS20_Training_{case}')
    try:
        gt = nib.load(os.path.join(path, f'BraTS20_Training_{case}_seg.nii')).get_fdata()
        p = predictByPath(model, path, case)
        
        if p is None:
            print(f"Skipping evaluation for case {case} due to prediction error.")
            return
        
        gt[gt != eval_class] = 0
        gt[gt == eval_class] = 1
        resized_gt = cv2.resize(gt[:,:,slice_idx+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        
        plt.figure(figsize=(10, 5))
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(resized_gt, cmap="gray")
        axarr[0].title.set_text('Ground truth')
        axarr[1].imshow(p[slice_idx,:,:,eval_class], cmap="gray")
        axarr[1].title.set_text(f'Predicted class: {SEGMENT_CLASSES[eval_class]}')
        
        os.makedirs(PREDICTION_DIR, exist_ok=True)
        plot_path = os.path.join(PREDICTION_DIR, f'single_class_eval_case_{case}_class_{eval_class}_slice_{slice_idx}.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved single class evaluation for case {case} to {plot_path}")
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")

def plot_training_history(history_file):
    """Plot training history from CSV log."""
    history = pd.read_csv(history_file, sep=',', engine='python')
    
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    train_dice = history['dice_coef']
    val_dice = history['val_dice_coef']
    epoch = range(len(acc))
    
    plt.figure(figsize=(16, 8))
    fig, ax = plt.subplots(1, 3, figsize=(16, 8))
    
    ax[0].plot(epoch, acc, 'b', label='Training Accuracy')
    ax[0].plot(epoch, val_acc, 'r', label='Validation Accuracy')
    ax[0].legend()
    ax[0].set_title('Accuracy')
    
    ax[1].plot(epoch, loss, 'b', label='Training Loss')
    ax[1].plot(epoch, val_loss, 'r', label='Validation Loss')
    ax[1].legend()
    ax[1].set_title('Loss')
    
    ax[2].plot(epoch, train_dice, 'b', label='Training Dice Coef')
    ax[2].plot(epoch, val_dice, 'r', label='Validation Dice Coef')
    ax[2].legend()
    ax[2].set_title('Dice Coefficient')
    
    plot_path = os.path.join(VISUALIZATION_DIR, "training_history.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved training history plot to {plot_path}")

def find_best_weights(directory):
    """Find the weights file with the lowest validation loss."""
    pattern = re.compile(r'model_(\d+)-([\d.]+)\.weights.h5')  # Updated to .weights.h5
    best_loss = float('inf')
    best_file = None
    
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            epoch, val_loss = match.groups()
            val_loss = float(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_file = filename
    
    return best_file, best_loss

def main():
    print(f"Processing BraTS2020 dataset with {NUM_GPUS} device(s)...")
    os.makedirs(PREDICTION_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    os.makedirs(CODING_DIR, exist_ok=True) 
    
    # Split dataset
    train_ids, val_ids, test_ids = split_dataset()
    if train_ids is None:
        print("Dataset splitting failed.")
        return
    
    # Plot data distribution
    plot_data_distribution(train_ids, val_ids, test_ids)
    
    # Create data generators
    training_generator = DataGenerator(train_ids, batch_size=BATCH_SIZE)
    valid_generator = DataGenerator(val_ids, batch_size=BATCH_SIZE)
    test_generator = DataGenerator(test_ids, batch_size=BATCH_SIZE)
    
    # Visualize sample batch
    visualize_sample_batch(training_generator)
    
    # Build and compile model within strategy scope
    with strategy.scope():
        input_layer = Input((IMG_SIZE, IMG_SIZE, 2))
        model = build_unet(input_layer, 'he_normal', 0.2)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=[
                'accuracy',
                tf.keras.metrics.MeanIoU(num_classes=4),
                dice_coef,
                precision,
                sensitivity,
                specificity,
                dice_coef_necrotic,
                dice_coef_edema,
                dice_coef_enhancing
            ]
        )
    
    # Plot model architecture
    try:
        plot_model(
            model,
            to_file=os.path.join(VISUALIZATION_DIR, 'model_architecture.png'),
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=False,
            dpi=70
        )
        print(f"Saved model architecture to {VISUALIZATION_DIR}/model_architecture.png")
    except (AttributeError, ImportError, OSError) as e:
        print(f"Warning: Failed to plot model architecture due to {str(e)}. Ensure Graphviz and pydot are installed correctly. Continuing without model visualization.")
    
    # Define callbacks
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.000001, verbose=1),
        ModelCheckpoint(
            filepath=os.path.join(CODING_DIR, 'model_{epoch:02d}-{val_loss:.6f}.weights.h5'),  # Updated to .weights.h5
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=True
        ),
        CSVLogger(os.path.join(CODING_DIR, 'training.log'), separator=',', append=False)
    ]
    
    # Train model
    K.clear_session()
    history = model.fit(
        training_generator,
        epochs=25,
        steps_per_epoch=len(train_ids),
        callbacks=callbacks,
        validation_data=valid_generator
    )
    
    # Save model
    model.save(os.path.join(CODING_DIR, "model.h5"))
    print(f"Saved model to {CODING_DIR}/model.h5")
    
    # Plot training history
    plot_training_history(os.path.join(CODING_DIR, 'training.log'))
    
    # Find and load best weights
    best_weights_file, best_val_loss = find_best_weights(CODING_DIR)
    best_model = build_unet(input_layer, 'he_normal', 0.2)
    best_model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=[
            'accuracy',
            tf.keras.metrics.MeanIoU(num_classes=4),
            dice_coef,
            precision,
            sensitivity,
            specificity,
            dice_coef_necrotic,
            dice_coef_edema,
            dice_coef_enhancing
        ]
    )
    
    if best_weights_file:
        best_model.load_weights(os.path.join(CODING_DIR, best_weights_file))
        print(f"Loaded best weights from {CODING_DIR}/{best_weights_file} with validation loss {best_val_loss}")
    else:
        print("No checkpoint weights found. Using trained model for predictions.")
        best_model = model
    
    # Evaluate on test set
    results = best_model.evaluate(test_generator, batch_size=100, callbacks=callbacks)
    descriptions = [
        "Loss", "Accuracy", "MeanIOU", "Dice coefficient", "Precision",
        "Sensitivity", "Specificity", "Dice coef Necrotic", "Dice coef Edema", "Dice coef Enhancing"
    ]
    print("\nModel evaluation on the test set:")
    print("==================================")
    for metric, description in zip(results, descriptions):
        print(f"{description} : {round(metric, 4)}")
    
    # Visualize predictions for test cases
    for i in range(min(7, len(test_ids))):
        case = test_ids[i][-3:]
        showPredictsById(best_model, case, start_slice=60)
    
    # Visualize predicted segmentations
    cmap = mcolors.ListedColormap(['black', 'red', 'green', 'blue'])
    norm = mcolors.BoundaryNorm([0, 1, 2, 3], cmap.N)
    for slice_to_plot in [60, 65]:
        show_predicted_segmentations(best_model, test_ids, slice_to_plot, cmap, norm)
    
    # Evaluate single class
    case = test_ids[3][-3:] if len(test_ids) > 3 else test_ids[0][-3:]
    evaluate_single_class(best_model, case, eval_class=2, slice_idx=40)

if __name__ == "__main__":
    main()