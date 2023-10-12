import os
import pandas as pd
import torch
import cv2
import torch.nn.functional as F
from monai.transforms import Resize, RandCropByPosNegLabel
from monai.handlers import CheckpointSaver
from monai.config import print_config
from monai.networks.nets import EfficientNetBN
from monai.transforms import (
    Compose,
    LoadImage,
    AddChannel,
    ScaleIntensity,
    RandRotate,
    RandFlip,
    RandZoom,
    ToTensor,
)
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data import DataLoader, Dataset
from monai.inferers import SlidingWindowInferer
from monai.handlers import CheckpointSaver, StatsHandler
from monai.utils import set_determinism
from monai.data import CacheDataset

from utils.MyDataset import *
from utils.MeanDiceMetric import *

HEIGHT_IMAGE_SIZE = 512  # Set your desired height size
WIDTH_IMAGE_SIZE = 512   # Set your desired width size

# Set deterministic training for reproducibility
set_determinism(seed=0)

# Initialize MONAI configuration (optional)
print_config()

# Define the data directory and CSV file
csv_file = 'dataset/dataset_train.csv'

# Define transforms for data preprocessing
transforms = Compose([
    ScaleIntensity(),
    RandFlip(spatial_axis=0, prob=0.5),
    RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
    ToTensor(),
])
transforms = None
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    kfold = 5
    # Create a MONAI dataset with transforms
    train_dataset = MyDataset(csv_file=csv_file, istrain=True, kfold=kfold,transforms=transforms)
    val_dataset = MyDataset(csv_file=csv_file, istrain=False, kfold=kfold, transforms=transforms)
    # Wrap the datasets with CacheDataset
    train_dataset = CacheDataset(train_dataset, num_workers=2, cache_rate=1.0)
    val_dataset = CacheDataset(val_dataset, num_workers=1, cache_rate=1.0)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=1)
    
    # Create an EfficientNet model
    model = EfficientNetBN(model_name="efficientnet-b4", num_classes=2)

    # Create loss function and metrics
    loss_function = DiceLoss(sigmoid=True)
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Create inferer for sliding window inference
    inferer = SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=2)

    save_dir = './output'

    # Check if the directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # Create the directory if it doesn't exist

    # Check if you have write permission
    if not os.access(save_dir, os.W_OK):
        print(f"Error: You do not have write permission in {save_dir}.")

    #stats_handler = StatsHandler()

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create an instance of the custom MeanDiceMetric
    best_metric = -1
    # Training loop
    for epoch in range(2):  # Number of epochs
        model.train()
        #print("model: ",model)
        epoch_loss = 0.0
        for batch_data in train_loader:
            inputs, labels = batch_data['image'].to(device), batch_data['label'].to(device)
            # Reshape inputs to match the expected shape. Assuming inputs has shape [5, 1, 3, 512, 512]
            inputs = inputs.view(-1, 3, 512, 512)
            optimizer.zero_grad()
            outputs = model(inputs)
            # Apply sigmoid activation to the model's output
            outputs = torch.sigmoid(outputs)
            labels = [[1, 0] if label == 0 else [0, 1] for label in labels]
            labels = torch.tensor(labels)  # Convert labels to a tensor
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/10] Loss: {epoch_loss / len(train_loader)}")

        # Validation
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                inputs, labels = val_data['image'].to(device), val_data['label'].to(device)
                # Reshape inputs to match the expected shape
                inputs = inputs.view(-1, 3, 512, 512)
                val_outputs = model(inputs)  # Directly use the model for inference
                # Add a channel dimension to val_outputs
                val_outputs = val_outputs.unsqueeze(1)
                # Convert labels to a tensor
                labels = torch.tensor(labels)
                # Continue with your evaluation or metric calculation
                dice_metric(y_pred=val_outputs, y=labels.view(-1, 1))

        # Calculate validation dice score
        val_dice_score = dice_metric.aggregate().item()
        dice_metric.reset()
        print(f"Validation Dice Score: {val_dice_score}")
        # Update the best score and save the model if it's the best
        if val_dice_score > best_metric:
            best_metric = val_dice_score
            print(f"Best Dice Score: {best_metric}")
            checkpoint_filename = os.path.join('./output', 'best_model.pth')
            torch.save(model.state_dict(), checkpoint_filename)