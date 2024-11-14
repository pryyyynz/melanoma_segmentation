# A model for skin lesion segmentation in melanoma

**Welcome!** This repository contains the code and resources for implementing JAAL-Net, a deep learning-based segmentation model for skin lesion detection.

## Table of Contents
- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Getting Started](#getting-started)
- [Installation Requirements](#installation-requirements)
- [Dataset](#dataset)
- [Training](#training)
- [Results](#results)
- [Usage and Customization](#usage-and-customization)

## Overview

Skin lesion segmentation is critical for early melanoma diagnosis. JAAL-Net combines a **Local Fusion Network (LF-Net)** and a **Dual Attention Network (DA-Net)** to effectively segment skin lesions in dermoscopic images. The model has been tested on the ISBI2016 dataset, achieving high accuracy, specificity, and sensitivity.

**Key Features:**
- Joint attention mechanisms enhance feature extraction.
- Adversarial learning approach reduces segmentation errors.
- Model components (LF-Net and DA-Net) perform feature fusion and contextual refinement.

## Model Architecture

JAAL-Net consists of two primary components:
1. **Local Fusion Network (LF-Net)** - Acts as the generator, focusing on capturing lesion features through contour attention, dense connections, residual blocks, and depthwise convolutions.
2. **Dual Attention Network (DA-Net)** - Serves as the discriminator, applying position and channel attention to highlight informative regions and improve segmentation quality.

### Detailed Architecture
The LF-Net enhances feature extraction with:
- **Contour Attention Layer** for edge-focused learning.
- **DenseNet and ResNet Blocks** for capturing multiscale features.
- **DDSConv and DeConv** layers for efficient feature recycling.

The DA-Net improves segmentation accuracy using:
- **Position Attention Module** to capture spatial relationships.
- **Channel Attention Module** to focus on feature relevance across channels.


## Getting Started

Follow these steps to set up the repository and start training the JAAL-Net model.

### Installation Requirements

To install the requirements, you can run:
```bash
pip install -r requirements.txt
```

## Dataset

We use the [ISBI2016 dataset](https://challenge.isic-archive.com/data/) for training and testing JAAL-Net. This dataset contains 900 training images and 379 test images with binary masks.

1. Download the ISBI2016 dataset from the official [ISIC Archive](https://challenge.isic-archive.com/data/).
2. Unzip the dataset and place it in the `data/` directory.
3. Ensure the folder structure follows `data/train` and `data/test` format for ease of use.

### Preprocessing
Images are resized to 128x128 for compatibility with JAAL-Net. This can be done using OpenCV or any image processing tool.

```python
import cv2
img = cv2.resize(img, (128, 128))
```

## Training

To train JAAL-Net on your local machine, run the following command:
```bash
python src/train.py --epochs 300 --batch_size 16 --learning_rate 0.0001
```

### Important Training Parameters
- **epochs**: Total number of training epochs.
- **batch_size**: Set depending on available GPU memory.
- **learning_rate**: Default is 0.0001; adjust to improve training stability.

#### Model Checkpointing
The model will save checkpoints in `checkpoints/` every few epochs for recovery and evaluation. You can adjust checkpoint frequency in `train.py`.

### Training Optimizations
The following optimizations are applied in training:
- **Gradient Clipping** to stabilize training.
- **Learning Rate Adjustment** for smoother convergence.
- **Model Checkpointing** to save progress intermittently.


Evaluation metrics include:
- **Dice Similarity Coefficient (DSC)**
- **Intersection over Union (IoU)**
- **Accuracy**
- **Sensitivity**
- **Specificity**

## Results

Our best results on the ISBI2016 dataset (after 300 epochs) are as follows:
- **Dice Similarity Coefficient**: 91.53%
- **IoU**: 89.68%
- **Accuracy**: 97.43%
- **Sensitivity**: 93.65%
- **Specificity**: 91.94%


## Usage and Customization

### Custom Dataset
To use JAAL-Net with a custom dataset:
1. Ensure images are in `data/train` and `data/test`.
2. Adjust image resizing parameters in `src/train.py` as needed.

