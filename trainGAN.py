import matplotlib.pyplot as plt
import numpy as np
import torch
 
import torch.nn as nn
 
from monai.data import DataLoader

import torch.optim as optim
from torch.nn import Module, Sequential, Conv2d, ConvTranspose2d, LeakyReLU, BatchNorm2d, ReLU, Tanh, Sigmoid, BCELoss 
from monai.losses import DiceLoss
from monai.utils import set_determinism
from utils.MyDataset import *
from utils.MeanDiceMetric import *
from utils.utils import *
# Always good to check if gpu support available or not
dev = 'cuda:0' if torch.cuda.is_available() == True else 'cpu'
device = torch.device(dev)

HEIGHT_IMAGE_SIZE = 512  # Set your desired height size
WIDTH_IMAGE_SIZE = 512   # Set your desired width size

# Set deterministic training for reproducibility
set_determinism(seed=0)

# Define the data directory and CSV file
csv_file = 'dataset/dataset_train.csv'
# Define transforms for data preprocessing
transforms = None
#https://www.topbots.com/step-by-step-implementation-of-gans-part-2/
if __name__ == '__main__':
    kfold = 5
    # Create a MONAI dataset with transforms
    train_dataset = MyDataset(csv_file=csv_file, istrain=True, kfold=kfold,transforms=transforms)
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=2)

    # creating gen and disc
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    # initializing the weights
    netD.apply(init_weights)
    netG.apply(init_weights)

    # Setting up otimizers for both Generator and Discriminator 
    opt_D = optim.Adam(netD.parameters(), lr = 0.0002, betas= (0.5, 0.999))
    opt_G = optim.Adam(netG.parameters(), lr = 0.0002, betas= (0.5, 0.999))

    # Setting up the loss function - BCELoss (to check how far the predicted value is from real value)
    loss = DiceLoss(sigmoid=True)

    epoch_loss = 0.0
    for i, batch_data in enumerate(train_loader):
        images, labels = batch_data['image'].to(device), batch_data['label'].to(device)
        # Reshape inputs to match the expected shape. Assuming inputs has shape [5, 1, 3, 512, 512]
        images = images.view(-1, 3, 512, 512)
        # Loss on real images
        # clear the gradient
        opt_D.zero_grad() # set the gradients to 0 at start of each loop because gradients are accumulated on subsequent backward passes
        # compute the D model output
        yhat = netD(images.to(device)) # view(-1) reshapes a 4-d tensor of shape (2,1,1,1) to 1-d tensor with 2 values only
        # specify target labels or true labels
        target = torch.ones(len(labels), 1, dtype=torch.float, device=device)  # Add a dimension here

        # calculate loss
        print ("losss: ", yhat.shape, target.shape, labels.shape, images.shape)
        loss_real = loss(yhat, target)
        # calculate gradients -  or rather accumulation of gradients on loss tensor
        loss_real.backward()

        # Loss on fake images
        # generate batch of fake images using G
        # Step1: creating noise to be fed as input to G
        noise = torch.randn(len(labels), 100, 1, 1, device = device)
        # Step 2: feed noise to G to create a fake img (this will be reused when updating G)
        fake_img = netG(noise) 
        print("fake_img: ", fake_img.shape)
        # compute D model output on fake images
        yhat = netD(fake_img.to(device)) # .cuda() is essential because our input i.e. fake_img is on gpu but model isnt (runtimeError thrown); detach is imp: Basically, only track steps on your generator optimizer when training the generator, NOT the discriminator. 
        # specify target labels
        target = torch.zeros(len(labels),1, dtype=torch.float, device=device)
        # calculate loss
        loss_fake = loss(yhat, target)
        # calculate gradients
        loss_fake.backward()
        
        # total error on D
        loss_disc = loss_real + loss_fake
        
        # Update weights of D
        opt_D.step()

        ##########################
        #### Update Generator ####
        ##########################
        # clear gradient
        opt_G.zero_grad()
        # pass fake image through D
        yhat = netD(fake_img.to(device))
        # specify target variables - remember G wants D *to think* these are real images so label is 1
        target = torch.ones(len(labels), 1, dtype=torch.float, device=device)
        # calculate loss
        loss_gen = loss(yhat, target)
        # calculate gradients
        #loss_gen.backward()
        # update weights on G
        opt_G.step()

        ####################################
        #### Plot some Generator images ####
        ####################################
 
        # during every epoch, print images at every 10th iteration.
        if i% 5 == 0:
            # convert the fake images from (b_size, 3, 512, 512) to (b_size, 512, 512, 3) for plotting 
            img_plot = np.transpose(images.detach().numpy(), (0,2,3,1))
            plot_images(img_plot)
            print("********************")

        