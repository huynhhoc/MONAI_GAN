from monai.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
import cv2
HEIGHT_IMAGE_SIZE = 512  # Set your desired height size
WIDTH_IMAGE_SIZE = 512   # Set your desired width size

class MyDataset(Dataset):
    def __init__(self, csv_file, istrain = True, kfold = 5, transforms=None):
        self.data = pd.read_csv(csv_file)
        if istrain:
            self.data = self.data[self.data['kfold'] != kfold]
        else:
            self.data = self.data[self.data['kfold'] == kfold]
        
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 2]  # Use the correct column index (2 in this case)
        
        # Load the image using OpenCV
        img = cv2.imread(img_path)
        # Convert the image to a float32 numpy array
        img = img.astype(np.float32)

        # Check the number of channels and add or repeat channels if needed
        if img.shape[-1] != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert to RGB if it's grayscale

        # Apply MONAI transformations
        if self.transforms:
            img = self.transforms(img)
        
        img = img/255
        # Resize the image to the desired size while ensuring 3 channels (RGB)
        img = cv2.resize(img, (WIDTH_IMAGE_SIZE, HEIGHT_IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        # Transpose the image to the correct shape [C, H, W] (channels, height, width)
        #img = img.transpose(2, 0, 1)

        # Add a batch dimension to the image
        #img = torch.tensor(img).unsqueeze(0)  # [C, H, W] to [B, C, H, W]
        label = self.data.iloc[idx, 4]  # Assuming the label column is in the CSV file
        return {'image': img, 'label': label}



