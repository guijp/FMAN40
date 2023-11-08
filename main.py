import torch 
from autoencoders import SimpleAutoencoder
from utils import train_auto
import matplotlib.pyplot as plt
from dataset import WSIAEDataset

# Creating dataset
patch_size = 100
dataset = WSIAEDataset(['wsi/20PK 02736-7_10x.png'], patch_size=patch_size, overlap=0)
print(len(dataset))
loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = 32, shuffle = True)

# Initializing model
model = SimpleAutoencoder(channels_in=3)

# Training model with dataset
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 5e-2, weight_decay = 1e-8)
model = train_auto(loader, model, optimizer, loss_function, n_epochs=100)

fig, ax = plt.subplots(nrows=5, ncols=2)

for i in range(5):
  image_batch = next(iter(loader))   
  image = image_batch[0].reshape(1,3,patch_size,patch_size)

  # Reshape the array for plotting
  item = image[0,:,:,:].movedim(0,-1)
  ax[i][0].imshow(item)

  rec_item = model(image)
  rec_image = rec_item[0,:,:,:].movedim(0,-1).detach().numpy()
  ax[i][1].imshow(rec_image)

plt.show()