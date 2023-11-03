import torch
import torch.nn as nn
import torch.nn as nn
from monai.networks.nets import EfficientNetBN
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import Module, Sequential, Conv2d, ConvTranspose2d, LeakyReLU, BatchNorm2d, ReLU, Tanh, Sigmoid, BCELoss 

import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # The Generator class is responsible for creating synthetic images that resemble real data.
        # It does this by progressively increasing the spatial dimensions using ConvTranspose2d layers.

        self.gen = nn.Sequential(
            # Initial layer to increase spatial dimensions
            nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=4, stride=4, padding=0, bias=False),  # Increase stride
            # Output size: b_size, 512, 4, 4
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
 
            # ConvTranspose layers to upsample spatial dimensions
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=4, padding=0, bias=False),  # Increase stride
            # Output size: b_size, 256, 16, 16  # Adjusted output size
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
 
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=4, padding=0, bias=False),  # Increase stride
            # Output size: b_size, 128, 64, 64  # Adjusted output size
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=4, padding=0, bias=False),  # Increase stride
            # Output size: b_size, 64, 256, 256  # Adjusted output size
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),  # Final stride adjustment
            # Output size: b_size, 3, 512, 512
            nn.Tanh()
        )
 
    def forward(self, input):
        """
        Forward pass of the generator.

        Args:
            input (torch.Tensor): The input noise with shape (batch_size, 100, 1, 1).

        Returns:
            torch.Tensor: The generated synthetic images with shape (batch_size, 3, 512, 512).
        """
        return self.gen(input)

class Discriminator(Module):
    def __init__(self):
        super().__init__()
        self.dis = Sequential(
            # input is (3, 512, 512)
            Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            LeakyReLU(0.2, inplace=True),

            Conv2d(in_channels=32, out_channels=32 * 2, kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(32 * 2),
            LeakyReLU(0.2, inplace=True),

            Conv2d(in_channels=32 * 2, out_channels=32 * 4, kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(32 * 4),
            LeakyReLU(0.2, inplace=True),

            Conv2d(in_channels=32 * 4, out_channels=32 * 8, kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(32 * 8),
            LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(32 * 8 * 32 * 32, 1)  # Fully connected layer for binary classification
        self.sigmoid = Sigmoid()

    def forward(self, input):
        """
        Forward pass of the discriminator.

        Args:
            input (torch.Tensor): The input data with shape (batch_size, num_channels, height, width).

        Returns:
            torch.Tensor: The output prediction with shape (batch_size, 1). 
            This represents the discriminator's confidence in the input data being real (closer to 1) or fake (closer to 0).
        """
        features = self.dis(input)
        features = features.view(input.size(0), -1)  # Flatten the features
        output = self.fc(features)
        output = self.sigmoid(output)
        return output

def init_weights(m):
    if type(m) == ConvTranspose2d:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif type(m) == BatchNorm2d:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)
            
# plot images in a nxn grid
def plot_images(imgs, grid_size = 5):
    """
    imgs: vector containing all the numpy images
    grid_size: 2x2 or 5x5 grid containing images
    """
    fig = plt.figure(figsize = (8, 8))
    columns = rows = grid_size
    plt.title("Training Images")
    try:
        for i in range(1, columns * rows + 1):
            plt.subplot(rows, columns, i)  # Create a subplot
            plt.imshow(imgs[i])
            plt.axis("off")
    except:
        pass
    plt.show()
#----------------------------------------------------
def getLabel (labels):
    labels = [[1, 0] if label == 0 else [0, 1] for label in labels]
    labels = torch.tensor(labels)  # Convert labels to a tensor
    return labels

def sample_noise(batch_size, z_dim=512):
    """
    Generate random noise as input to the generator.

    Args:
        batch_size (int): The number of noise samples to generate.
        z_dim (int): The dimension of the latent space.

    Returns:
        noise (torch.Tensor): A batch of random noise samples.
    """
    noise = torch.randn(batch_size, z_dim)
    return noise
