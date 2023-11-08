import torch 
import torch.nn.functional as F

class SimpleAutoencoder(torch.nn.Module):
    def __init__(self, channels_in):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(channels_in, 8, 3, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8, 16, 3, padding=1),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Flatten(),
            torch.nn.Linear(int(100*100*16/4), 10)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(10, int(100*100*16/4)),
            torch.nn.Unflatten(1, torch.Size([16, 50, 50])),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(16, 8, 3, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8, channels_in, 3, padding=1)
        )

    def forward(self, x):
        x_enc = self.encoder(x)
        x_dec = self.decoder(x_enc)
        return x_dec
