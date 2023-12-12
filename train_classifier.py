import matplotlib.pyplot as plt
from dataset import WSIAEDataset
from autoencoders import Transfer, SSIMLoss, Classifier
import numpy as np
from PIL import Image
import torch
import numpy as np
from datasets import get_datasets
import sys

# Settings
sys.stdout = open(sys.stdout.fileno(), 'w', 1)
loss_function_bce = torch.nn.BCELoss().cuda()
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("running on: {}".format(device))

# Creating dataset
train_paths = [
               'Gleason/Data_1_SUS/20181120_SUS/train_299',
               'Gleason/Data_0_SUS/20180911_SUS/train_299',
               'Gleason/Data_0_SUS/20180911_Helsingborg/train_299',
               'Gleason/Data_0_SUS/20180911_Linkoping/train_299',
               'Gleason/Data_0_SUS/20180911_Rotterdam/train_299'
              ]
tr_dataset, val_dataset = get_datasets(train_paths, device, val_split=0, binary=True)
loader_trn = torch.utils.data.DataLoader(dataset=tr_dataset, batch_size=32, shuffle=True)
n_tr = len(tr_dataset)

# Initializing classifier
classifier = Classifier().cuda()

# Training
optimizer = torch.optim.Adam(classifier.parameters(), lr = 1e-3, weight_decay = 1e-8)
n_epochs = 1

# Training model
for epoch_i in range(n_epochs):
    print(f' -> epoch_i: {epoch_i+1} / {n_epochs}')
    i=1
    total_images = 0
    for image, target in loader_trn: 
        image = image.to(device)
        total_images += image.shape[0]
        target = target.to(device).reshape(-1,1).float()
    
        y_pred = classifier(image)
        loss = loss_function_bce(y_pred, target)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i%10==0:
            print(f'[{total_images}/{n_tr}] | loss: {loss} | accuracy: {(target == (y_pred>0.5)).float().mean()}')
        i+=1
    torch.save(classifier, 'classifier.pht')
    
torch.save(classifier, 'classifier.pht')
