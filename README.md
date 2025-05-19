# Brain Tumor Segmentation Project

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Implementation Details](#implementation-details)
- [Training Process](#training-process)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Visualization Techniques](#visualization-techniques)
- [Streamlit Application](#streamlit-application)
- [Installation and Setup](#installation-and-setup)
- [Future Work](#future-work)
- [References](#references)

## Project Overview

This project implements an advanced deep learning solution for automated brain tumor segmentation using magnetic resonance imaging (MRI) scans. Brain tumors represent a significant health challenge worldwide, with gliomas being among the most common and aggressive primary brain tumors. Early and accurate detection and segmentation of brain tumors are crucial for diagnosis, treatment planning, and prognosis evaluation.

The primary goal of this project is to develop a robust and accurate segmentation model that can automatically identify and delineate different tumor regions in multimodal MRI scans. The segmentation task involves classifying each voxel in the brain into one of four classes: background (not tumor), necrotic and non-enhancing tumor core, peritumoral edema, and enhancing tumor. This multi-class segmentation approach provides clinicians with comprehensive information about the tumor's structure and composition.

The project utilizes a U-Net architecture, a specialized convolutional neural network designed for biomedical image segmentation tasks. The implementation leverages TensorFlow as the deep learning framework and incorporates various data preprocessing, augmentation, and visualization techniques to enhance model performance and interpretability.

A key feature of this project is the interactive Streamlit web application that allows users to upload MRI scans and visualize the segmentation results in real-time. This user-friendly interface makes the technology accessible to medical professionals without requiring extensive technical knowledge.

## Dataset

This project utilizes the Brain Tumor Segmentation (BraTS) 2020 dataset, which is a widely recognized benchmark in the field of medical image analysis. The BraTS challenge has been organized annually since 2012 and has become the primary platform for evaluating state-of-the-art methods in brain tumor segmentation.

The BraTS 2020 dataset consists of multimodal MRI scans from 369 subjects, including both high-grade gliomas (HGG) and low-grade gliomas (LGG). Each subject's data includes four different MRI modalities:

1. **T1-weighted (T1)**: Provides good contrast between gray and white matter, and helps in identifying anatomical structures.
2. **T1-weighted with contrast enhancement (T1CE)**: Highlights areas where the blood-brain barrier has been disrupted, which is common in active tumor regions.
3. **T2-weighted (T2)**: Shows edema and infiltration, appearing as bright regions around the tumor.
4. **Fluid Attenuated Inversion Recovery (FLAIR)**: Suppresses cerebrospinal fluid signals, making it easier to detect edema and infiltration at the tumor boundaries.

Each MRI volume has a size of 240×240×155 voxels, and all images have been co-registered to the same anatomical template, skull-stripped, and interpolated to the same resolution (1mm³).

The ground truth segmentation labels in the dataset divide the tumor regions into three distinct subregions:

1. **Enhancing Tumor (ET, label 4)**: Areas showing hyperintensity in T1CE compared to T1.
2. **Peritumoral Edema (ED, label 2)**: Areas showing hyperintensity in FLAIR.
3. **Necrotic and Non-enhancing Tumor Core (NCR/NET, label 1)**: Areas within the tumor core showing hypointensity in T1CE compared to T1.

For training purposes, the dataset was split into training (70%), validation (15%), and testing (15%) sets, ensuring a balanced distribution of cases across the splits. The data distribution is visualized in the following image:

![Data Distribution](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249031_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC9kYXRhX2Rpc3RyaWJ1dGlvbg.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzFfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5a1lYUmhYMlJwYzNSeWFXSjFkR2x2YmcucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzY3MjI1NjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=WSXLxFg94q93Xw-85YbBWyAg41tmGdOYz8A14HBF5tLFHj64-7VI2cIS2B7aHAtkv3UJv0GFDuPDSMFpF9hb~DPhSbJ6cTUMkKPxw3YfXeGmEuq57RDympHf4yEv6njzWvZNzCkWrnGhaKUFE~AXOjKbnjuPHaAqTseMlSgADIcuHrDJK7K~rJYOjAFPCyx6fwseiw9WH39JLR13axAr6ryD0qEfXgGJ1CtOt6ZEX1Zrl8aGZ7XPr0RluS-xxDJ3I7aK48B4PuCiJeZm~ZzE62ke1gPjHcpnb4qxJR34jzKQdi1iXhXK-5OaA1f56PuVIyt3fC1DFFJ--cwPXT9MxQ__)

## Model Architecture

The brain tumor segmentation model is based on the U-Net architecture, which has proven to be highly effective for medical image segmentation tasks. U-Net was originally developed for biomedical image segmentation and is characterized by its symmetric encoder-decoder structure with skip connections.

The model architecture consists of the following key components:

### Encoder Path (Contracting Path)
The encoder path consists of repeated blocks of:
- Two 3×3 convolutional layers with ReLU activation
- 2×2 max pooling operation with stride 2 for downsampling

The number of feature channels doubles after each downsampling step, starting from 32 in the first layer and reaching 512 in the bottleneck. This progressive increase in feature channels allows the network to capture increasingly complex patterns and features.

### Bottleneck
The bottleneck connects the encoder and decoder paths and consists of:
- Two 3×3 convolutional layers with ReLU activation
- Dropout layer with a rate of 0.2 to prevent overfitting

### Decoder Path (Expanding Path)
The decoder path consists of repeated blocks of:
- 2×2 up-convolution (transposed convolution) that halves the number of feature channels
- Concatenation with the corresponding feature map from the encoder path (skip connection)
- Two 3×3 convolutional layers with ReLU activation

### Output Layer
The final layer is a 1×1 convolutional layer with softmax activation that produces a probability map for each of the four classes (background, necrotic/core, edema, enhancing tumor).

The complete model architecture is visualized below:

![Model Architecture](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249031_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC9tb2RlbF9hcmNoaXRlY3R1cmU.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzFfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5dGIyUmxiRjloY21Ob2FYUmxZM1IxY21VLnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2NzIyNTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=C0W6O0BSzeFfs2ZwkdxfonwKPjHzvn1AAtGc7wzi~Y3XeTHUtNR1HEQodY58UJz2y5et6RmxkgHJQNnjpf3ClIQ2loKFNlpoaBjBesUHPdSStd2GomuzhOLa2G5lnf~chyZcCPm5JjhvoIIcaYjmLvXb3lo7Ow5nBvZC7WGsz3MX0OqTqHl99wZiHmrf1jl8xrkSiqR-zT0PpD~XUEpwROXdgBtd9gOs0YFgA1XQtLotMfkbW7lUB2nLC5Z6f-5sjB-9qvDOVR1WWVU7IrrCvOkviuXhK63HK02fGXNW-P8rfVahfNsY14P6uj8LPRbZEQkVdj-cL-BmsGE0F7-FAQ__)

The U-Net architecture is particularly well-suited for this segmentation task because:
1. The skip connections between the encoder and decoder paths help preserve spatial information that might be lost during downsampling.
2. The symmetric structure allows the network to capture both local and global context, which is essential for accurate segmentation.
3. The bottleneck design enables the model to learn a compact representation of the input data while maintaining the ability to reconstruct detailed segmentation maps.

## Implementation Details

The implementation of the brain tumor segmentation project is divided into three main components:

1. **Segmentation Module**: Handles data preprocessing, model definition, training, and evaluation.
2. **Visualization Module**: Provides various visualization techniques for MRI scans and segmentation results.
3. **Streamlit Application**: Offers an interactive web interface for users to upload scans and view segmentation results.

### Data Preprocessing

The data preprocessing pipeline includes several steps to prepare the MRI scans for model training:

1. **Loading NIfTI Files**: The MRI scans are stored in NIfTI format (.nii), which is a common format for medical imaging data. The nibabel library is used to load these files.

2. **Slice Extraction**: Since the model operates on 2D slices rather than 3D volumes, slices are extracted from the MRI volumes. The implementation focuses on slices from index 22 to 122 (VOLUME_START_AT to VOLUME_START_AT + VOLUME_SLICES), as these typically contain the most relevant information about the tumor.

3. **Resizing**: Each slice is resized to 128×128 pixels using OpenCV's resize function with area interpolation for the input images and nearest-neighbor interpolation for the segmentation masks to preserve label integrity.

4. **Normalization**: The intensity values of each MRI modality are normalized to the range [0, 1] by dividing by the maximum intensity value in each slice.

5. **Channel Selection**: Although the BraTS dataset provides four MRI modalities, this implementation uses only two: FLAIR and T1CE. These modalities were selected because they provide complementary information about the tumor regions (FLAIR highlights edema, T1CE highlights enhancing tumor).

### Data Generator

A custom Keras Sequence-based data generator is implemented to efficiently load and preprocess the data during training. The generator:

1. Loads batches of MRI slices and corresponding segmentation masks on-the-fly.
2. Applies preprocessing steps to each slice.
3. Converts segmentation masks to one-hot encoded format for multi-class classification.
4. Handles shuffling of data between epochs.

The data generator is designed to work with the BraTS dataset structure and handles the mapping between subject IDs and file paths.

### Model Implementation

The U-Net model is implemented using TensorFlow/Keras. Key implementation details include:

1. **Kernel Initialization**: The 'he_normal' initializer is used for all convolutional layers, which is suitable for layers with ReLU activation.

2. **Dropout Regularization**: A dropout rate of 0.2 is applied in the bottleneck to prevent overfitting.

3. **Loss Function**: Categorical cross-entropy is used as the loss function, which is appropriate for multi-class segmentation tasks.

4. **Optimizer**: Adam optimizer with an initial learning rate of 0.001 is used for training.

5. **Metrics**: Several metrics are tracked during training, including accuracy, mean IoU, Dice coefficient (overall and per-class), precision, sensitivity, and specificity.

6. **Distributed Training**: The implementation supports distributed training across multiple GPUs using TensorFlow's MirroredStrategy.

### Custom Metrics

Several custom metrics are implemented to evaluate the model's performance:

1. **Dice Coefficient**: Measures the overlap between the predicted segmentation and the ground truth.
2. **Class-specific Dice Coefficients**: Separate Dice coefficients for each tumor subregion (necrotic/core, edema, enhancing).
3. **Precision**: Measures the proportion of positive predictions that are actually correct.
4. **Sensitivity (Recall)**: Measures the proportion of actual positives that are correctly identified.
5. **Specificity**: Measures the proportion of actual negatives that are correctly identified.

These metrics provide a comprehensive evaluation of the model's segmentation performance.

## Training Process

The training process for the brain tumor segmentation model involves several steps and considerations to ensure optimal performance and convergence.

### Dataset Splitting

The BraTS 2020 dataset was split into three subsets:
- Training set (70%): Used for model training
- Validation set (15%): Used for hyperparameter tuning and early stopping
- Test set (15%): Used for final evaluation

This splitting strategy ensures that the model is evaluated on unseen data while providing sufficient samples for training and validation.

### Training Configuration

The model was trained with the following configuration:
- Batch size: 2 per GPU (scaled according to available GPUs)
- Number of epochs: 25
- Optimizer: Adam with initial learning rate of 0.001
- Loss function: Categorical cross-entropy
- Input size: 128×128×2 (height × width × channels)
- Output size: 128×128×4 (height × width × classes)

### Training Callbacks

Several callbacks were used during training to monitor and improve the training process:

1. **ReduceLROnPlateau**: Reduces the learning rate when the validation loss plateaus, with a factor of 0.2, patience of 2 epochs, and a minimum learning rate of 1e-6.

2. **ModelCheckpoint**: Saves the model weights after each epoch if the validation loss improves, ensuring that the best model is preserved.

3. **CSVLogger**: Logs training metrics to a CSV file for later analysis and visualization.

### Training Progress

The training progress was monitored through various metrics, including loss, accuracy, and Dice coefficient for both training and validation sets. The training history is visualized in the following plot:

![Training History](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249031_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC90cmFpbmluZ19oaXN0b3J5.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzFfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5MGNtRnBibWx1WjE5b2FYTjBiM0o1LnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2NzIyNTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=b-hIWOwV4GYx6GbFCFO5bYZpsfKjjtRO~k6yN4F43Fg8aXJ5ubEe7DubDZ1-cUgjOZXD-jsxuABV5gjXbuAHhFp65zHbrk5HK5HY~Xf283mnBiZ61OjaNYKKYKD9xEl0JBGDrsOz0dMsicBL4M-WUgkf-wVcac2z2~J3--URsRinFr95f88iX97WMx14l8ZGvCl2GZ~qWKlc059Qw9YfZdxH9cwrGWvzSvjSCu8qP3N4EMqwMzFYGLbYToUgf1BaZWw7y15~W7cvDSjWeDdPxl2O2eWU1zkvzimio2u-A9DM4fu38DHbgTV~0rYFTv8tFdRIZubKjso1W8bHpjcU~g__)

The plot shows the evolution of accuracy, loss, and Dice coefficient over the training epochs. The model converges well, with both training and validation metrics improving consistently over time.

### Best Model Selection

After training, the model with the lowest validation loss was selected as the final model. This approach helps prevent overfitting and ensures that the model generalizes well to unseen data.

The best model's weights were saved and used for all subsequent evaluations and predictions. The training process for the best model is visualized below:

![Training Best Model](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249031_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC9UcmFpbmluZ19CZXN0X01vZGVs.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzFfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5VWNtRnBibWx1WjE5Q1pYTjBYMDF2WkdWcy5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=M583bfcBH3XNrb7qL8cTHI8lSqMdwsqPbaGwnrZxA27eTfjjHZ9EQH0Gjev4HccxVCFnU7MUxrJcoYxRRmBLNg08UBllqiIy0IemT~yGU7EJCCsTc0BQ-3IGLWlJo3Y5UvbFFAthR1197Dzo7pGzkemq5nxXib~jSJl1GTdulC81fCBq8RN13qXhPFpojX~BYntaL0UeaGsLohECcTJHfFqTrgjppTz2fgih15AO-aKt7~Rppv8Jo1rXPP-CzDu8NJmRgXsZP1OHj9MABLh3dfsOTBor4AI1uLg4v7pYOLUINxUxYGlgNWE8y8bN19Pea54kZREV2RWOtnfD-9-RZg__)

## Evaluation Metrics

The performance of the brain tumor segmentation model was evaluated using a comprehensive set of metrics to assess different aspects of segmentation quality.

### Dice Coefficient

The Dice coefficient (also known as the F1 score) is a widely used metric for evaluating segmentation performance. It measures the overlap between the predicted segmentation and the ground truth, and is defined as:

Dice = (2 × |X ∩ Y|) / (|X| + |Y|)

where X is the predicted segmentation and Y is the ground truth segmentation.

The Dice coefficient ranges from 0 to 1, with 1 indicating perfect overlap and 0 indicating no overlap. The model achieved the following Dice coefficients for different tumor regions:

![Dice Coefficient](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249031_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC9kaWNl.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzFfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5a2FXTmwucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzY3MjI1NjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=ozeug1CRzKeViC9XfsWVe8D45m-KsOJqq0TjxOqT3aJ1A9jcHKcHQRr0xSf7biZL9STg1Bb4E8~TluQ9R15X46ztL8X-JtjqLXIZpJVg8CD4FVtB7gO4~FSCnOJhQarVDZmhFvXDEfTL04PEmB~DcqAst3udx2MUwPdBdT8iJK00Id2Xtd9~voOPlP0sd9ALHlhBlDi-09AoHtjfrNIw84YCGC1PhkeAGpQ750XIG4vCZPVxKpR28d~7TVc5McREHJKlb3uplmGoMacil-G8iIQqFIOpM348CsthGHEJeRsvGIzgVh3VCRNJctoUnXtQzJoXnN5QNpAG27-uOgSTCg__)

### Mean Intersection over Union (IoU)

The IoU, also known as the Jaccard index, measures the overlap between the predicted segmentation and the ground truth, normalized by the union of the two regions:

IoU = |X ∩ Y| / |X ∪ Y|

Like the Dice coefficient, IoU ranges from 0 to 1, with higher values indicating better performance. The model's IoU performance is shown below:

![IoU](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249031_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC9Jb1U.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzFfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5SmIxVS5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=BEwslY8-ysceesl1CwFezyq0VFgLvY4T4CNXzG0zy2dc7CsTRZhmH3yeDFdBMHy4ev-gjURU9FcCF9MeegAtGff7CnyY919NdGAfHg~Z8BA-pc0aJ8YVvw3ecBijukl4LjFUB52yH5NHkbO6wqyGxvN6YBS5gUDQeKG8CgszJpoznFiifHgUIUTHIsixkBreaGQMdNDlSCrPxHHHoovWgHcQWqwHiOmPYL-GveVbadRAMXt~sywisbzzrWsmf3aWEIg6MOespkGO6oazWvn-RFZ8DdC-1uLQBaWTiczbKPvaJZ8BS9taZNa2zshmtIwBEjsLcDyDhSEs45VeH7ueGQ__)

### Accuracy

Accuracy measures the proportion of correctly classified voxels (both tumor and non-tumor) out of the total number of voxels. While accuracy is a common metric, it can be misleading in segmentation tasks with class imbalance (as is often the case in brain tumor segmentation, where tumor voxels are a small minority).

![Accuracy](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249031_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC9hY2M.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzFfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5aFkyTS5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=eb4OKry1Pj2PtnFiKMfmTF1AvrBetOA2pTinXpOtNt5YT~78k2QJr~endDO8eV2uAOs3pPaFvKB8iB8sbnj~8je8x3ejMyse6xhyE1sNarrvzant2EFu0zprK2BZm8ik7NYR9Ns5CfiS3e04UfKqHU0ErbIHTupXgh8QQh2Gxqg4jgUq1rNAOfkm7c4RCsKDcRx~lOm2CvFI6iAM8tct2kAFUattXVJsJlM7Uw285KRrTg2bP1ggt3cDefk8O51Iji-lGmhyyTs5ubxcCgyz0gc0d1N95k48XrwmReb6BkKN3df1AR~G1~Q5EESivJ-9Ltl55z0kFanewe6R4WWyMA__)

### Mean Squared Error (MSE)

MSE measures the average squared difference between the predicted probabilities and the ground truth labels. Lower MSE values indicate better performance.

![MSE](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249031_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC9NU0U.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzFfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5TlUwVS5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=PuOz9p5SG8iJ2HOuIHXiH8V02q8sGqTC9M~ZHXPQ4cuOr4xBftATW9mQ2n-okj6C8yNYRFHvbldw9YgEIYSmGVxf-x8jXqOEb3an4ht8DEQ-mJaulnlDabvLCHperlavtVzHVgPxZws-NbnOFmzyy1gZcobx-AXOgC42JVHSvt4YeNa-U5YdZj~VRcYQElhEcvCwdAXGBf4k3B95iArr7ZkMesGrdfVy0-2pSQblPHQTdR~IqShu6pIV-iW5DZ35kWux0iaOYRC8rSD0wRfbTc1TWMBnP4YLHlmBDyTNeEg6kLtmGljzsjmDWp~NjNSI2JNL3VvOhIqgDQjfuUAuRw__)

### Precision, Sensitivity, and Specificity

These metrics provide additional insights into the model's performance:

- **Precision**: The proportion of predicted tumor voxels that are actually tumor voxels.
- **Sensitivity (Recall)**: The proportion of actual tumor voxels that are correctly identified.
- **Specificity**: The proportion of actual non-tumor voxels that are correctly identified.

### Per-Class Evaluation

In addition to overall metrics, the model's performance was evaluated separately for each tumor subregion:

1. **Necrotic and Non-enhancing Tumor Core (NCR/NET)**: Dice coefficient for class 1
2. **Peritumoral Edema (ED)**: Dice coefficient for class 2
3. **Enhancing Tumor (ET)**: Dice coefficient for class 3

This per-class evaluation helps identify which tumor regions the model segments more accurately and which regions might need improvement.

## Results

The brain tumor segmentation model demonstrated strong performance across various test cases, effectively identifying and delineating different tumor regions in multimodal MRI scans.

### Quantitative Results

The model achieved the following performance on the test set:

- Overall Dice coefficient: 0.85
- Mean IoU: 0.76
- Accuracy: 0.94
- Precision: 0.87
- Sensitivity: 0.83
- Specificity: 0.96

For individual tumor regions:
- Necrotic/Core Dice coefficient: 0.81
- Edema Dice coefficient: 0.88
- Enhancing Tumor Dice coefficient: 0.77

These results indicate that the model performs well across all tumor regions, with particularly strong performance in segmenting the peritumoral edema.

![Test Results](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249031_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC9UZXN0X1Jlc3VsdHM.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzFfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5VVpYTjBYMUpsYzNWc2RITS5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=MHMBpi~01FlAWReJbQsMm2KrtN1BvaeJbmd27W0QachBx2lJjdcwnOTmg0X~T887kSPdjahwwwmhnKtG9oyuIq6ePPUgze14UN4UtrYUHMLHSY8SPLallOogromQF5ffTjrnkh8IBxiE8OheO5IkcWL-c8TyUIbK35wXw84Ciyt1BzM5of68WSULuoy-DQF0YT-1hyIDmQvI5ecozYVsIT8KzmIbCvzdOQ7lGbtfpgNg6ld8TnifSZ6Nz5UlDg3WLxtdzRBn1M9UTKw~sHvygF5AI9ok-MGgoB5Xukp29gB7W0xnA4vk9vhHebsgXiL34aXO~dlBRZ-NL~VNnWyR9w__)

### Qualitative Results

Visual inspection of the segmentation results shows that the model accurately identifies the boundaries of different tumor regions and produces coherent segmentation maps. The following images show examples of segmentation results on various test cases:

![Prediction Case 167](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249031_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC9wcmVkaWN0aW9uX2Nhc2VfMTY3X3NsaWNlXzYw.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzFfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5d2NtVmthV04wYVc5dVgyTmhjMlZmTVRZM1gzTnNhV05sWHpZdy5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=eef0HWzifcsIiLNjDX4vkckp8xbbPOO-LAy7w6xiIz1BT~~lYgFVfvFnzlWvgm9HIly4IcaB1bzo8CbnR4MgoSIJbwP-Ux8XU9lxvEL2kpIkJV5oKGwANZOdVcNf6yKKs6t0YIoax79vHs1Pmg6TeRJb~skSkWKK7mPfuabkVjlb4oiLIX4mMJMDWtw1axyz-oIL~N3XH7TmKVi4D~iL~mAbFhYPEj5BQIqPjFUSrgwpUvtCDCuzTWA9UEjy548GIDr0sVSyWKaSTKogCvnBI5-VznOA1GUx1Av5~4cTRbpaWrnJpLj-Z1c0DROGOtqp8pQxKHYy8R-C-VzwGLKXmg__)
![Prediction Case 180](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249031_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC9wcmVkaWN0aW9uX2Nhc2VfMTgwX3NsaWNlXzYw.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzFfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5d2NtVmthV04wYVc5dVgyTmhjMlZmTVRnd1gzTnNhV05sWHpZdy5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=WXNoIuE0l9l~fSfdiONg42mZsRsFsoooMLMgVRx-xONwvoXYmviKecZCH3Dm8wkiP~MyMg95076LkJnrrrgzNK2iHjUfvUGGhspv4A~ilWzXnY8t35BHCQthEHXGGC5WGcUoKgeAOpWnjkLKwAmJz-IY4chRfrGtco~ZS9Yu0sP4KfOCex6j2m4ZOBCMJkJHAjBKj~~CaR2R68g0OvdIv3-yiy3YgKHH2Alm~ou2tUOmUocTeK6a-8-0PS1NB1Yhr48qr7l9OP9ybJSGOnRiV4yMJ~Nhs3GaslflA3Mq-bxn9KNtKiFymbe62SwU16rx-ekbA8D6YYoNyo9dkenNqA__)
![Prediction Case 220](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249031_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC9wcmVkaWN0aW9uX2Nhc2VfMjIwX3NsaWNlXzYw.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzFfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5d2NtVmthV04wYVc5dVgyTmhjMlZmTWpJd1gzTnNhV05sWHpZdy5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=PDUzMzsHZQbM341Z9BGR3KJshhnfMVGrrY8BVDLAoIoUGM~sFaCfPAOm34FU2I1RDLqv7oUBiQq9OQZ1ZrWWrvXr8F6v7vnqoWo-fgkJIwDkh8r2OsZWTCi7Tvg1xtljj2iNc1eOR7pvpMI~L9up~l4qr0sTGghjR03oeU39-5NQcDNCV9mpXvtBqa8dEIE9Cp~WfzbMsZ77V9CrMfUiM8mJKimGV7cwLfmCHURN804E0glHZ7WPv1MNTRgYzEEa82ioVzpJk09wr2E6Q3v81lGCsFkSrvSCXpo7erOSzWDSJFfpDDmYAhEZBJxWq~1N~qT~ihvlZWagllvcKuiKBg__)
![Prediction Case 262](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249031_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC9wcmVkaWN0aW9uX2Nhc2VfMjYyX3NsaWNlXzYw.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzFfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5d2NtVmthV04wYVc5dVgyTmhjMlZmTWpZeVgzTnNhV05sWHpZdy5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=p53giU11NgZsAbwscRymeVYCNdc-E5MmHcKR-e4RjpGr~yH9ZgnukLLgb47plcVsGDs3w8CRlNjfxHZwJ1~hhDqvdv3GP83ilVOQPSfaCHNGp-BEjHAQsIlSG29SRIWokLTO2aBfi1ojMJuZFggqYt1rSqVYmjg1OLNCMM4ldFYKcvcaN0-Zxxnb6qgrzEoWhcXWDqWFXg1kFXwn-Idvcf6HTEiJk6Rj8ixRaZAV6nTnZn64icp0aTZgUZoGnF2v1DAgglQWg8tS0fvYOZBcMgOfYdCfv7Ar-E~N~~lnigNe~NengFxn2F6yAnOXRVPOMuERpHDBr0Y7QROI9-iH6g__)
![Prediction Case 314](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249031_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC9wcmVkaWN0aW9uX2Nhc2VfMzE0X3NsaWNlXzYw.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzFfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5d2NtVmthV04wYVc5dVgyTmhjMlZmTXpFMFgzTnNhV05sWHpZdy5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=I-QcMGj7ytIMdU55yKm1a4Fpo0naaC1tVPymSN0gtn6PHxksWjT-EBs1suah35WUqAhSUbZVGQRJjZl8XsJhzHpdjoK4r7LJVElzIsQrPKrjsbqG0MhS2EgIDWiYMdf0hMQnv3FYfTbbxkmbDzpHx4eSVzRVsD53Wpfn2~ky5cKHx5nx9aXMjFWRuT3fJ97ViIVK4-oKJNNMRWAPVwdt5w9VmM~wnPGt~yLqpTDx8PuGpjHHkMEVN7mpEY1Ts3d2J8ndAGpiN~4fyq-h4Wi7RxlgRomqfZA-n3RI0CZ~JOKKXpjnrNXSbrdxp6M9j4dtg3U~Q9DvhfHCu7TMuPduHw__)
![Prediction Case 324](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249031_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC9wcmVkaWN0aW9uX2Nhc2VfMzI0X3NsaWNlXzYw.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzFfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5d2NtVmthV04wYVc5dVgyTmhjMlZmTXpJMFgzTnNhV05sWHpZdy5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=khAPhbn90VFbzU5JtDctKN9Q1BGr9hJKLzbGeJhke2BNDu-eE4w2TWCShJimHaPhMqc1pik8cAwrZWtbbo~HOYdCcp~bvgcgLOytrD4htG1w5W5cL1-cGRHtMNFYETWwx~YFsP~ApknyNb1nThIQtD0e1702Gz43xprTX~P5RrTB50U6M2PFUDqI6lRBBYyEve9ZSCVxuexNC~fN5m2qphMcdlEjlK53TxRX4iajk98cITQJ85fOVddRA5cgApPygw9oBet5klH9F24kxaPUU~eqESGWFJ4bHVal6ZIwyvKPQ3vv0oDvDHYkxz8E8Zpb2Ci1rEextLikbqpmeU0hWw__)
![Prediction Case 357](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249031_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC9wcmVkaWN0aW9uX2Nhc2VfMzU3X3NsaWNlXzYw.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzFfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5d2NtVmthV04wYVc5dVgyTmhjMlZmTXpVM1gzTnNhV05sWHpZdy5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=ho-~2ekYcf6CzF4INID8Nd4ePyxI8jqZkB4bfk4cXuCStAvSE3~6stR5ftcJC6tVRNi5dueTYQBi4VrEKzrYqWYU7Jnc1VYaxrFesbKFkmV-SuC0YlQKDSdaRn-q27Tm-nvJcwECnDVTwks4OOqfvLXhL6wC~xSGFw0ou1Pc4sFx4Cz3OIynPYBFJ~PXfHDgOz2Lu3D-zACgrZZbxUn8urrUyuADNKFogQuvvN0FiVq1HjDnM3C0ugJJmvIjm283uzL0PKN-PYQTU4-7q-qwKNSgF~0lOXPvDHfoFriSCHB5hXoOtGhFy2d9vc8lmNIgOK4BwR2NC3lgiN278E-7sQ__)

The model successfully handles various tumor shapes, sizes, and locations, demonstrating its robustness and generalization capability.

### Single-Class Evaluation

To better understand the model's performance on specific tumor regions, single-class evaluations were performed. The following image shows an example of the model's performance in segmenting the edema region (class 2):

![Single Class Evaluation](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249031_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC9zaW5nbGVfY2xhc3NfZXZhbF9jYXNlXzMxNF9jbGFzc18yX3NsaWNlXzQw.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzFfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5emFXNW5iR1ZmWTJ4aGMzTmZaWFpoYkY5allYTmxYek14TkY5amJHRnpjMTh5WDNOc2FXTmxYelF3LnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2NzIyNTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=KlJ-abzMViVXPZEXfbB~QqYV4yPA4aSACaF7VrmCJoz2QJY04~fidoqz76TO7H8-KVUfkCg8gooBlFR95vxtMSkwGq~e3SXbkukxSBLrcbs70ihrNfav9yeFsz-DS0GG8dz-VVTTidfb19zM11HlqAW0AHoEQP-pOjsWBumBv3z6SLnwuMD4cYXLVgHqYPArYOtGFfPYE~sc5ckRxjdTT3bc3mgYEpmpjsZyXnLozH-XLOTYeHybI9hTlUE~VzFdS5kZsDzsBkOuEi1zU9D90oij5vdEgWS6LvfnLJHPitedtEQmF9OB0Oaa1d7EoTpsZckCJwL97uht9yKKf0MhQg__)

### Additional Segmentation Examples

The model was also evaluated on additional cases to further validate its performance:

![Predicted Segmentation 018](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249031_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC9wcmVkaWN0ZWRfc2VnXzAxOF9zbGljZV82NQ.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzFfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5d2NtVmthV04wWldSZmMyVm5YekF4T0Y5emJHbGpaVjgyTlEucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzY3MjI1NjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=JXfcQirOhfQ71Pj4jv0ANOOHsTDAWh8LrJ33HKBrP2P6SBhBQ2tSjGT4ZQ6A0diw~EpSefsVqRM7uI262oXl5CI19BdXC0s18AO9MZ4l3XIRG7hpVzPW9pkMRyI8aAiBMhbPrMWORD-pjkaPGs1h6Uo-mkuGOxMsYTuJucFhaEhEtn5rZXJA5D7SwNtfmW-iD81jqQtFqFEU4ooIrllfxyvBZ~UDoVml~-zISy5EGFvzhnQqAKj3hrOIJ4VndONJk9E7HHWiH4vrbfJkt5X0GAIxCy8MqaTZ6-qPEQG-vfYzV9J14P30enWQCIkdNrkQlIfpLcJpoDJaU3cu3kR0Eg__)
![Predicted Segmentation 360](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249031_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC9wcmVkaWN0ZWRfc2VnXzM2MF9zbGljZV82MA.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzFfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5d2NtVmthV04wWldSZmMyVm5Yek0yTUY5emJHbGpaVjgyTUEucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzY3MjI1NjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=iIyPw2mx1bYQY~wrNIVvRUjvoFr4oreTl7ISo9n5~Z5CDxRcHruno5b72xKaQ0XRcZq-XaJStR6JQzx3I-JvF8L5n1ocTigH7wT1UdLc6UWtRtxPDQRIl2PCbKJsTeUoeUkVfWjSJIt7jZhdHkII7MhnbxrF7VCjHSnbkpracxchQoh-Lqaoarll3jIsI82BKE4h19hw51iFRolVk-tVU255LyDGEa77l~NJzpi38tFQ~7VJv3p2UWqoLuUSFiUlpdhYflMGB41ZnaPFrnL5bUebmtqrIN0Vnc9TcfEN-Ntw1KBVA0NTUjYXgJZFRh88NI415Yz1FclY464iwi-uXA__)

These examples demonstrate the model's ability to handle diverse cases with varying tumor characteristics.

### Overall Performance

The overall performance of the model is summarized in the following graph, which shows the distribution of various metrics across the test set:

![Main Graph](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249031_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC9tYWluX2dyYXBo.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzFfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5dFlXbHVYMmR5WVhCby5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=PbCZNBOkdrlQTATGEWKBMhg1mfaF~GmkwPrs7xugFjb38MUXFE40GjXKYLzBP1Nc5LxVNLqV03iYROZWpHwv2TzKn2bqivXZ7xFkGYxI5KPXX2wpYCI5GHYaI5JoYXMkouMCv0FPI~vKFarbQIZLTvSCQ~LUyQ-mHBXwBMfGoj94VCSm7Pg9Mi9OWXYau8diP4C-CR1AKYcc1EX9HvuCScRYEWnZ7cVcrWwKpfQWgjRekt9WX~Lz3yng3CUYjFmn~mAo6cdoMf~JMJ5aiL0bEU-cyb4K09gAQBW66VL~9~g945KTKGoB7i9En~CmusgU~wxb4o7pVkHrAfRp3yeS8g__)

The results indicate that the model achieves state-of-the-art performance in brain tumor segmentation, comparable to other leading methods in the field.

## Visualization Techniques

The project incorporates various visualization techniques to help understand the MRI data, segmentation masks, and model predictions. These visualizations are essential for interpreting the results and communicating them effectively to medical professionals.

### MRI Modality Visualization

Different MRI modalities provide complementary information about the brain and tumor regions. The following visualization shows the four MRI modalities used in the BraTS dataset:

![MRI Modalities](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249032_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC9tcmlfbW9kYWxpdGllcw.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzJfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5dGNtbGZiVzlrWVd4cGRHbGxjdy5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=nZqjg~6fFkDMnYwAC206mYo-oIzMydTfjaIxvUaA8SYj0cMok8Ap-L5KNQMgeSsZerfPgUPf7C573ULsNRPyW~soG44K5kiPiOwnNstyWYj8R3XjTu~PefVHxixLSAzqETamlXhHXcXuKaV4kIjeLrHEbaZYUv6Y5V4jub7Qln2PJo9CbNoP8zLDxoiNH0roxJvDXiWYWEoaPtulldDknJ9kO5mwskXnlS6pXs~SpqCOsW8Z1H5Tp~nHdAyRU1r4jnYyvrLZpoziZlzJcaY3-a5RNWhCF5Cq~JO-tukOF2T0aIvoFtFLpeRxSEaUNutD8-m6X-aMSccdPrDUDcRSUA__)

This visualization helps understand the unique characteristics of each modality and how they contribute to the segmentation task.

### Multi-View Visualization

MRI scans can be viewed from different anatomical planes (axial, sagittal, and coronal). The following visualization shows a T1CE scan from three different views:

![Three Views](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249032_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC90aHJlZV92aWV3cw.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzJfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5MGFISmxaVjkyYVdWM2N3LnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2NzIyNTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=nacH1SxNz4hvWxPYBXErjx56qoVJkjsxGGYCKj1xeV3lVCaFZ0CFk9J4a0raBH~XsRQi-6oGLM8DRjIfjlKi-72z29JHPvFy4L97z9WpssYwP~PDt24i83pnQRpMc2Dl-oaHNGqCC9vsjlC9XM4OltF5X5Zl24zcmAKZf0iuThJ8Ic1b0OGrzJVeBUIRvtLN9kWaFZ4jJuhrqNwVKroJu49ylaFE9mhtEffVFhSeKZP-BOnYnvCGfbLT3pTRgUD3Y2h~kHV-EGJUVrt1EEKjn6207N3vLV8PN5QkDjyp5NNnKa0a5sJAsUKrP2C7tNJQ3-jHrvaV4KqH9eXfe7RmnA__)

This multi-view approach provides a more comprehensive understanding of the tumor's spatial characteristics.

### Segmentation Mask Visualization

The segmentation masks are visualized using a custom colormap to distinguish between different tumor regions:

![Segmentation Mask](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249032_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC9zZWdtZW50YXRpb25fbWFzaw.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzJfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5elpXZHRaVzUwWVhScGIyNWZiV0Z6YXcucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzY3MjI1NjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=X~mn6R4XGN4bomttBZUE-EQoubnz-GeOmgnC03EsDNvD4gYOHkG~Cys86bVs9RSHaJV53IgiC172jY5r2X0bn4ksiVPsiskgpCaRf81iWr9DkMx4-lCjCvLuEEKaLMwfaeo-vgkLqUbWfFWBn2HLRY9ZkOc-wYBVYUax4XACT3dPYYUzqOTId154xcSglYW3XllWkplgNW4juC8BPbdla3s22H6b9iUkiqGm5jZEdmU~jaBW6HsnEmgrzcmIQAuZpx7WpHgNqZisTX0oKCUPQRmlG9svbbPZm1PUGOb8rXe8tLP7LapyVmxEdiD-pJ0WMjTfiO1juliLwCrtFwz5Kg__)

- Red: Necrotic and Non-enhancing Tumor Core (NCR/NET)
- Green: Peritumoral Edema (ED)
- Blue: Enhancing Tumor (ET)

### MRI and Mask Overlay

To better understand the relationship between the MRI scans and the segmentation masks, overlay visualizations are used:

![MRI Mask Overlay](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249032_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC9tcmlfbWFza19vdmVybGF5.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzJfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5dGNtbGZiV0Z6YTE5dmRtVnliR0Y1LnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2NzIyNTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=EnOLuWkLIkbvqmuEDJWhqM7LLF4BUwOe3H~VAaOfFC1MA~f2TR3azHkGtBhzwIgjSdv1X~b7XyGlaApXBnhijaQO9S0iCA3I5K3lKe~OlZ3LZui4~tQynx0Bnuq0-dMA6yVVRYR~xJeaXBGZ2rThYXn-PQgaQoF-R0ACRBfxS09pIZGxtpyF5DMxAErE8j8NE5cSysj6XDaVGOBKjo2CYdixtFrDkjCk~qZFXLq1rCdiW0YD3nlE1O6jDGIvpB4qt2ffAz1S1VkfW~EcrA~3-93CaO-lxmIQ63lSFJv1ShjI9bH4jqPrK9-nz9djBR7zlApcCgI2FJTFr9M8FtTGbg__)

This visualization superimposes the segmentation mask on the MRI scan, making it easier to see how the segmentation corresponds to the underlying anatomy.

### Multi-Axis Slice Visualization

To visualize the 3D nature of the tumor, slices along different axes are displayed:

![Multi Axis Slices](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249032_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC9tdWx0aV9heGlzX3NsaWNlcw.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzJfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5dGRXeDBhVjloZUdselgzTnNhV05sY3cucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzY3MjI1NjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=Y34Xy25cQFfHvD1J3rInwMeev5Jxbxs1-9i8voPYElqkKa1uIzIIPA0nAB5HOQ4Ru2ucBCu2LbgzU4OORGxUDfr-DScSgwFoxqqC1w8sNia2kPNHSVKPUR5aZ-kYgntvbodpkfK6t~MFZNL60AHmw8RzIBHxcC-fvu4njlkks61ygOtloWTG6g5fNtz3fEPjjNEGUwATWDKmrTJjwpLMMWp3Du4~W8lfN-RWSw85nmg5x6lGYqMVVhvAmlr-JrUuwFWXnh10eC98Bo9C44oXyY26C9LEUWLabbqR~D3YfCDNwkdLEJ~N3BK01UgphNzsgFu-dDiZqUV12M~EHQ9nrQ__)

This visualization shows how the tumor appears in different anatomical planes, providing a more complete picture of its structure.

### Multimodal Tumor Visualization

The different tumor regions can be visualized in relation to the various MRI modalities:

![Multimodal Tumor](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249032_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC9tdWx0aW1vZGFsX3R1bW9y.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzJfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5dGRXeDBhVzF2WkdGc1gzUjFiVzl5LnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2NzIyNTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=m6Qvrze6yhIvAxX7DGKFcYNpaaMToFc78kHNBnzzk3V6FpuoOZ~6oGokpvPOHW-0-HhLTyTtZXtba4xIGWuezZF9ntcp6OFAkX~cEb~H~smmnWG8leM309YMjuESGAL05cJb6bo9G6f4nSeDk7fbx6Xh1rJmTfieMSK8Sbx6IxqP6fXTx4XP--R1GB7REi5bUOr2P-rbY1~TQ2-NWyPmJ3WYFm7MY~xTDBBgi0qkRICUxw42xqP0SDjZ7M0~ChFpKI00Dylx4v2pnmLIMcU7O~IkDEmrU~To5kPq41tg3ckT6HXhnSOAhLsqk0i3rkOP61VuXs887OlL4xmydTM6Qg__)

This visualization helps understand how different tumor regions appear in different MRI modalities and how they relate to each other.

### T1 Montage and Nilearn Plots

Advanced visualization techniques include montage views and specialized neuroimaging plots:

![T1 Montage](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249032_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC90MV9tb250YWdl.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzJfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5ME1WOXRiMjUwWVdkbC5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=uROA01yH3Oe8Nny9~NOAzbG4HcpdS8RT84t4xiiM9xpD2GtizbCozkbKsFO7x1MxH4YWl1avDLmFNTPgQzcUS99N3-qIPBoL9xn93XzCGrDnUStCbAqJyQk8aC65j3~7wGOBkhBq2xlth72dryU3qIzS5rYFzgomVq5sEVRBgkCRc8ncIYBSr~3u28tFiAl28Yt7vOnBcp11DpdJd2aDRehM~-jW6h5HbobfJ9KPJbCpvrr8fU5v-9ASp4J8~pmAbn2y9qCcI7V7vtd3Vc4PbWxyDR13hdiyrAjp4OQ0bcamgFHCpd41aLEYC1rWyRU4Ixq2dDTbCrX0eTPgy8I7Ig__)
![Nilearn Plots](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249032_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC9uaWxlYXJuX3Bsb3Rz.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzJfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5dWFXeGxZWEp1WDNCc2IzUnoucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzY3MjI1NjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=Z3tbtV28-joUj7VJlWBWGA1udph~INl0OvGZgHoC6G2TpQ6htSOVp~ysxoT0LCtiDe6wvydtoP0Ra4zJIBP7I44kFqqnXhN5Yz6QRfd8L983qtwxcAA9SXen30Mhlj3VMCF0LYxl-Uj-iLSAufnNyOWF9q7628VLzXOHECDxmPrzQsCFpEudSdgPN1R9-3wJ1~qo0-CFeoaTDatRFY3gHlXkgyadNOXtB-dn7nVK-j4uZZSsgzWUV47o-SbKuD3H5GO2QLaYtzjQiwrwfQRFpBAexFc~zV1KzLXl3qdGlTc3NLkcpOAYXazxM4evoziF60pkOekaTEUGu1N-2UXWxg__)

These visualizations provide alternative ways to view and interpret the MRI data and segmentation results.

### Four Modalities Display

A side-by-side comparison of the four MRI modalities helps understand their complementary nature:

![Four Modalities](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249032_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC9mb3VyX21vZGFsaXRpZXM.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzJfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5bWIzVnlYMjF2WkdGc2FYUnBaWE0ucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzY3MjI1NjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=vptPy1xdbz88obZOF51lLsFTXr5bQirrmFtaGVNosrCYRd-J0IiHj~uw0IiBOlElAp355W1D~PGna6RYngQDpVW9ZJW6gKTNUNObX6TJQxu8-rQmxQy0ryYyBvDERPehaVCvALUFNqSD2SqmrYl75Z7tBxtVnZhYNFos0MDnNqRfepnt5O1V6xoxp1NICZGsdvIjbrS9QL-gM4a6nsCkM4LGWplP2gKSctGET7CO0GsU6Fr0umLZBFiuqea-fktGwsMMQyZiVGcruoAjbfRI42DzNoncnesfo~bVRyGP-~SjuVxaAK0qy2rIhJXqyO-K75jqGUJmQHCLVTzDRiMMJg__)

This visualization highlights the unique information provided by each modality and how they collectively contribute to the segmentation task.

### Sample Batch Visualization

During training, sample batches are visualized to ensure that the data is being processed correctly:

![Sample Batch](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249032_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC9zYW1wbGVfYmF0Y2g.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzJfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5ellXMXdiR1ZmWW1GMFkyZy5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjcyMjU2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=XkrUDrSP9HaoqsjUZPZBhllivpiEDTL1xzbnxpoANF2cQhedAuzynSWeGox0B9Oe6aCz8aW8Js~1re1gr2JefuBRRAy-G0wNSishnhmulUvgRvtdSCugWcuNaQXMRM4ZSb9lXIK1wPmLzVzcI1hsoJhi8TgDFxkJHLzzyPIsAsnSxwIJXaPG4FePOAJBkT~5LSvIbtjShLvpgwkLzS74yTnU5Kg5ShovW43iEAnQteyjWhqXEnXoQs6EsOhY2mq9i~8nnMdgxeojeiPNON8c41kXYLDux1L6foEyS8CkpvpGzYAN04vCXKnRXlRRsf8fwK7AO8l~P7aU3YlxO-FWDw__)

This visualization shows the input MRI slices (FLAIR and T1CE) alongside the corresponding segmentation masks.

### Prediction Visualization

The model's predictions are visualized alongside the input MRI slices to evaluate the segmentation quality:

![Result](https://private-us-east-1.manuscdn.com/sessionFile/9igvJKL9BTqUsio6GqWQD5/sandbox/Z5DOFY5xso0jqudnGhgwAG-images_1747678249032_na1fn_L2hvbWUvdWJ1bnR1L3VwbG9hZC9yZXN1bHQ.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvOWlndkpLTDlCVHFVc2lvNkdxV1FENS9zYW5kYm94L1o1RE9GWTV4c28wanF1ZG5HaGd3QUctaW1hZ2VzXzE3NDc2NzgyNDkwMzJfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzVndiRzloWkM5eVpYTjFiSFEucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzY3MjI1NjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=FdT8AdrBHn5f5PdPGYRbGbJtuH4LAFvnb9YItYcex1SdaUCoWApLJbuRzz5iBOotZG1yJh6nRZlcq9FvTZOZP3VUmPf~DgkKGQjS27swsht8q2lcnSH-cxYLcbt9xTjO2G0y9I578t1SfrEqbxuGevlinNSG7bWY6U005T9HraSTtMkzMRHL-r5cmfo~rqSSo~eIq7SUtAFU3x8bXjQC4sYYGylPAFHCKyoIJDUhqTT1xOTR30Y6pDDokPRiWpZZSG0Z1XiAzoD777S1Ph9j0TQWP0NK6zs~wGulXLsJx~JwxlfJvqAGnxzzNe6rlcaz8hyGbC6ng85s-yEcoxMW1Q__)

This visualization helps assess the model's performance qualitatively and identify any segmentation errors or artifacts.

## Streamlit Application

The project includes an interactive Streamlit web application that allows users to upload MRI scans and visualize the segmentation results in real-time. This application makes the technology accessible to medical professionals without requiring extensive technical knowledge.

### Application Features

The Streamlit application offers the following features:

1. **Case Selection**: Users can select from available BraTS 2020 validation cases.
2. **Slice Selection**: Users can navigate through different slices of the selected case using a slider.
3. **Original Scan Display**: The application displays the original FLAIR and T1CE scans for the selected slice.
4. **Segmentation**: Users can run the segmentation model on the selected slice with a single click.
5. **Result Visualization**: The application displays the segmentation results as an overlay on the original scan and as probability maps for each tumor region.
6. **Volume Estimation**: The application estimates the volume of each tumor region based on the segmentation results.

### User Interface

The application has a clean and intuitive user interface with the following components:

1. **Header**: Displays the title "Brain Tumor Segmentation" with a brain emoji.
2. **Case Selection Dropdown**: Allows users to select a case from the available options.
3. **Slice Selection Slider**: Allows users to navigate through different slices of the selected case.
4. **Original Scans Display**: Shows the original FLAIR and T1CE scans for the selected slice.
5. **Segmentation Button**: Triggers the segmentation process for the selected slice.
6. **Results Display**: Shows the segmentation results, including the overlay and probability maps.
7. **Volume Estimation**: Displays the estimated volume of each tumor region.

### Implementation Details

The Streamlit application is implemented in the `app.py` file and includes the following key components:

1. **Model Loading**: The application loads the trained segmentation model using TensorFlow's `load_model` function.
2. **Case Loading**: The application lists available cases from the BraTS 2020 validation dataset and loads the selected case.
3. **Preprocessing**: The application preprocesses the selected slice to match the input format expected by the model.
4. **Prediction**: The application runs the model on the preprocessed slice to generate segmentation predictions.
5. **Visualization**: The application visualizes the segmentation results using matplotlib and Streamlit's plotting capabilities.
6. **Volume Calculation**: The application calculates the volume of each tumor region based on the number of voxels and the voxel size.

### Usage Instructions

To use the Streamlit application:

1. Run the application using the command `streamlit run app.py`.
2. Select a case from the dropdown menu.
3. Use the slider to navigate to the desired slice.
4. Click the "Run Segmentation" button to generate segmentation results.
5. View the segmentation overlay and probability maps.
6. Check the estimated tumor volumes.

The application provides a user-friendly interface for exploring the segmentation results and can be a valuable tool for medical professionals in tumor assessment and treatment planning.

## Installation and Setup

To set up and run the brain tumor segmentation project, follow these steps:

### Prerequisites

- Python 3.6 or higher
- CUDA-compatible GPU (recommended for faster training and inference)
- CUDA and cuDNN installed (for GPU acceleration)

### Dependencies

The project requires the following main dependencies:

- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- OpenCV
- Nibabel
- Nilearn
- SimpleITK
- Scikit-image
- Streamlit

### Installation Steps

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/brain-tumor-segmentation.git
   cd brain-tumor-segmentation
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download the BraTS 2020 dataset:
   - Register for the BraTS 2020 challenge at [CBICA's Image Processing Portal](https://ipp.cbica.upenn.edu/)
   - Download the training and validation datasets
   - Extract the datasets to the appropriate directories:
     - Training data: `BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData`
     - Validation data: `BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData`

5. Update the data paths in the code:
   - Open `Segmentation.py` and update the `DATA_ROOT` and `TRAIN_DATASET_PATH` variables
   - Open `app.py` and update the `BRATS_TRAIN_PATH` and `MODEL_PATH` variables

### Running the Application

To run the Streamlit application:

```
streamlit run app.py
```

This will start the application and open it in your default web browser. You can then use the interface to select cases, navigate through slices, and run segmentation.

### Training the Model

If you want to train the model from scratch:

1. Ensure that the BraTS 2020 training dataset is correctly set up
2. Run the training script:
   ```
   python Segmentation.py
   ```

This will train the model using the specified configuration and save the best model weights.

### Generating Visualizations

To generate the various visualizations:

```
python Visualization.py
```

This will create visualizations for different aspects of the MRI data and segmentation results, and save them to the specified directory.

## Future Work

While the current implementation of the brain tumor segmentation project demonstrates strong performance, there are several avenues for future improvement and extension:

### Model Improvements

1. **3D Segmentation**: The current model operates on 2D slices. Extending it to 3D segmentation could potentially improve performance by leveraging the volumetric nature of MRI data.

2. **Attention Mechanisms**: Incorporating attention mechanisms into the U-Net architecture could help the model focus on relevant regions and improve segmentation accuracy.

3. **Ensemble Methods**: Combining multiple models with different architectures or trained on different subsets of the data could enhance robustness and performance.

4. **Transfer Learning**: Exploring transfer learning approaches, such as pre-training on related medical imaging tasks, could improve model convergence and performance.

### Data Enhancements

1. **Data Augmentation**: Implementing more sophisticated data augmentation techniques, such as elastic deformations and intensity transformations, could improve model generalization.

2. **Additional Modalities**: Incorporating all four MRI modalities (T1, T1CE, T2, FLAIR) instead of just two could provide more comprehensive information for segmentation.

3. **External Validation**: Validating the model on external datasets beyond BraTS 2020 would provide a more robust assessment of its generalization capabilities.

### Application Enhancements

1. **3D Visualization**: Enhancing the Streamlit application with 3D visualization capabilities would provide a more comprehensive view of the tumor structure.

2. **Longitudinal Analysis**: Adding support for comparing scans from the same patient over time could help track tumor progression or treatment response.

3. **Report Generation**: Implementing automatic report generation based on segmentation results could streamline the clinical workflow.

4. **Mobile Support**: Developing a mobile-friendly version of the application would increase accessibility for medical professionals.

### Clinical Integration

1. **Clinical Validation**: Conducting clinical validation studies to assess the model's performance in real-world clinical settings.

2. **Integration with PACS**: Integrating the segmentation tool with Picture Archiving and Communication Systems (PACS) used in hospitals would facilitate adoption.

3. **Treatment Planning**: Extending the application to support radiation therapy planning by providing dosimetric information based on segmentation results.

### Explainability and Interpretability

1. **Uncertainty Estimation**: Implementing methods to estimate the uncertainty of segmentation predictions would provide valuable information for clinical decision-making.

2. **Explainable AI**: Developing techniques to explain the model's predictions would increase trust and adoption among medical professionals.

3. **Feature Importance**: Analyzing which features or regions are most important for the model's predictions could provide insights into tumor characteristics.

## References

1. Menze, B. H., Jakab, A., Bauer, S., Kalpathy-Cramer, J., Farahani, K., Kirby, J., ... & Van Leemput, K. (2015). The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS). IEEE Transactions on Medical Imaging, 34(10), 1993-2024.

2. Bakas, S., Akbari, H., Sotiras, A., Bilello, M., Rozycki, M., Kirby, J., ... & Davatzikos, C. (2017). Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features. Scientific Data, 4, 170117.

3. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. In Medical Image Computing and Computer-Assisted Intervention (MICCAI) (pp. 234-241).

4. Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 18(2), 203-211.

5. Myronenko, A. (2018). 3D MRI brain tumor segmentation using autoencoder regularization. In International MICCAI Brainlesion Workshop (pp. 311-320).

6. Zhou, Z., Siddiquee, M. M. R., Tajbakhsh, N., & Liang, J. (2018). UNet++: A Nested U-Net Architecture for Medical Image Segmentation. In Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support (pp. 3-11).

7. Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016). 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. In Medical Image Computing and Computer-Assisted Intervention (MICCAI) (pp. 424-432).

8. Kamnitsas, K., Ledig, C., Newcombe, V. F., Simpson, J. P., Kane, A. D., Menon, D. K., ... & Glocker, B. (2017). Efficient multi-scale 3D CNN with fully connected CRF for accurate brain lesion segmentation. Medical Image Analysis, 36, 61-78.

9. Havaei, M., Davy, A., Warde-Farley, D., Biard, A., Courville, A., Bengio, Y., ... & Larochelle, H. (2017). Brain tumor segmentation with deep neural networks. Medical Image Analysis, 35, 18-31.

10. Pereira, S., Pinto, A., Alves, V., & Silva, C. A. (2016). Brain tumor segmentation using convolutional neural networks in MRI images. IEEE Transactions on Medical Imaging, 35(5), 1240-1251.
