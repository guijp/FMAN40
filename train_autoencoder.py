import matplotlib.pyplot as plt
from utils.model_definitions import Transfer, SSIMLoss
import numpy as np
from PIL import Image
import torch
import numpy as np
from utils.datasets import get_datasets
import sys

# Settings
sys.stdout = open(sys.stdout.fileno(), 'w', 1) # enabling nohup run with real-time logs
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
Image.MAX_IMAGE_PIXELS = None
loss_function_ssim = SSIMLoss().to(device)
loss_function_mse = torch.nn.MSELoss().to(device)
loss_function_mae = torch.nn.L1Loss().to(device)
print("running on: {}".format(device))

# Creating dataset
train_paths = [
               'Gleason/Data_1_SUS/20181120_SUS/train_299',
               'Gleason/Data_0_SUS/20180911_SUS/train_299',
               'Gleason/Data_0_SUS/20180911_Helsingborg/train_299',
               'Gleason/Data_0_SUS/20180911_Linkoping/train_299',
               'Gleason/Data_0_SUS/20180911_Rotterdam/train_299'
              ]

tr_dataset, val_dataset = get_datasets(train_paths, device, val_split=0.1, binary=True)
n_tr = len(tr_dataset)
loader_trn = torch.utils.data.DataLoader(dataset=tr_dataset, batch_size=1, shuffle=True)
n_tr = len(tr_dataset)

# # Initializing model
n_auto=4
autoencoder = Transfer(n=n_auto).to(device)
# autoencoder = torch.load('autoencoder.pht')
# n_auto = len(autoencoder.encoders)

# Training model
optimizer = torch.optim.Adam(autoencoder.parameters(), lr = 1e-3, weight_decay = 1e-8)
n_epochs = 1

for dim in range(n_auto): # For each possible latent space size
    print(f'dim: {dim}')
    for epoch_i in range(n_epochs):
        total_images = 0
        i=0
        for image, target in loader_trn: 
            image = image.to(device)
        
            reconstructed = autoencoder(image, dim)
    
            loss_ssim = loss_function_ssim(reconstructed, image)
            loss_mse = loss_function_mse(reconstructed, image)
            loss_mae = loss_function_mae(reconstructed, image)
            loss = 0.5*loss_ssim + 0.5*loss_mse + 0.5*loss_mae
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_images += image.shape[0]
            if i%100==0:
                print(f'[{total_images}/{n_tr}] - loss: {loss}')
            i+=1
        torch.save(autoencoder, 'autoencoder.pht')
        print(f' -> epoch_i: {epoch_i+1} / {n_epochs}')
        
torch.save(autoencoder, 'autoencoder.pht')
