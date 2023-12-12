import torch
from piqa import SSIM
import numpy as np
import datetime

class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)

class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = torch.nn.Sequential(
            # 256 x 256 x 3
            
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.MaxPool2d(2, 2),

            # 128 x 128 x 16

            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(2, 2),

            # 64 x 64 x 32

            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(2, 2),

            # 32 x 32 x 64

            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(2, 2),

            # 16 x 16 x 128

            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.MaxPool2d(2, 2),

            # 8 x 8 x 256

            torch.nn.Conv2d(256, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(512),
            torch.nn.MaxPool2d(2, 2),

            # 4 x 4 x 512

            torch.nn.Conv2d(512, 1024, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(1024),
            torch.nn.MaxPool2d(2, 2),

            # 2 x 2 x 1024

            torch.nn.Conv2d(1024, 2048, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(2048),
            torch.nn.MaxPool2d(2, 2),

            # 1 x 1 x 2048
        )

        self.dense_layers = torch.nn.Sequential(
            # 1 x 1 x 2048

            torch.nn.Flatten(),

            # 2048
            
            torch.nn.Linear(2048, 100),
            torch.nn.Tanh(),
            
            # 100
            
            torch.nn.Linear(100,10),
            torch.nn.Tanh(),

             # 10
            torch.nn.Linear(10,1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x_enc = self.conv_layers(x)
        y_pred = self.dense_layers(x_enc)
        return y_pred
        
class Transfer(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.info = {'created_at': datetime.datetime.utcnow() ,
                     'n': n,
                     'size_training':[0]*n}
        
        self.encoders = torch.nn.ModuleList()
        self.decoders = torch.nn.ModuleList()

        # 0, 1, 2, ..., n-1
        for i in np.arange(0,n):
            self.encoders.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(3*2**(i), 3*2**(i+1), 3, padding=1), 
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(3*2**(i+1)),
                    torch.nn.MaxPool2d(2, 2),
                )
            )
            
            temp = []
            for j in np.arange(i,-1,-1):
                temp.append(torch.nn.Upsample(scale_factor=2))
                temp.append(torch.nn.Conv2d(3*2**(j+1), 3*2**j, 3, padding=1))
                temp.append(torch.nn.Sigmoid() if j==0 else torch.nn.ReLU())
            self.decoders.append(torch.nn.Sequential(*temp))

    def encode(self, x, n):
        for i in range(n+1):
            x = self.encoders[i](x)
        return x

    def decode(self, x_enc, n):
        x_dec = self.decoders[n](x_enc)
        return x_dec
        
    def forward(self, x, dim):
        if dim==-1:
            return x
        else:
            for i in range(dim):
                self.encoders[i].requires_grad_(False)
                x = self.encoders[i](x)
                
            self.encoders[dim].requires_grad_(True)    
            x = self.encoders[dim](x)
            
            x = self.decoders[dim](x)

            self.info['size_training'][dim] += x.shape[0] 
            
            return x
