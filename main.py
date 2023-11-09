import torch 
from autoencoders import SimpleAutoencoder
import matplotlib.pyplot as plt
from dataset import WSIAEDataset

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("running on: {}".format(device))

# Creating dataset
patch_size = 100
dataset = WSIAEDataset(['wsis_2023-11-03/20PK 02736-7_10x.png'], patch_size=patch_size, overlap=0)
loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = 32, shuffle = True)

# Initializing model
model = SimpleAutoencoder(channels_in=3).to(device)

# Training model with dataset
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 5e-2, weight_decay = 1e-8)

n_epochs=1
for _ in range(n_epochs):
  for image in loader: 

    reconstructed = model(image)
    loss = loss_function(reconstructed, image)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()  

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

plt.savefig('results.png')