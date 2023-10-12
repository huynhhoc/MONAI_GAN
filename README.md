# Medical Image Segmentation with EfficientNet and MONAI

This repository contains code for training a medical image segmentation model using the EfficientNet architecture and the MONAI library. The model is trained for binary segmentation, with the goal of segmenting specific structures or regions of interest in medical images.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Dataset](#dataset)
- [Usage](#usage)
- [Training](#training)
- [Inference](#inference)

## Prerequisites

Before you begin, make sure you have the following dependencies installed:

- Python 3.x
- PyTorch
- MONAI
- OpenCV
- Other required libraries (dependencies listed in the code)

You can install MONAI and its dependencies using pip:

```
pip install monai

```

# Dataset

The dataset used for training and validation is expected to be provided as a CSV file containing information about image file paths and corresponding labels. Ensure that you have the dataset in the appropriate format and specify the path to the CSV file in the script.
Usage

# Data Preparation

Before running the training script, you should collect your own medical image dataset. To generate the required CSV files for training and testing, you can use the data_preparation.py script provided in this repository:

```
python data_preparation.py

```

This script will generate dataset/dataset_train.csv and dataset/dataset_test.csv based on your collected data.

Clone the repository to your local machine:

```

git clone https://github.com/huynhhoc/dlMONAIexample.git
cd dlMONAIexample

```
# Training

You can start training the segmentation model by running the provided script:

```
python train.py

```

The script performs the following steps:

    Data preprocessing and augmentation using MONAI transforms.
    Model architecture setup (EfficientNet with binary classification).
    Loss function and evaluation metric definition (Dice Loss and Dice Metric).
    Model training with Adam optimizer.
    Validation and best model checkpointing based on the Dice score.

You can customize various training settings in the script, such as batch size, learning rate, and the number of epochs.

# Inference

Once the model is trained, you can perform inference on new medical images. Modify the script as needed for your specific inference requirements.