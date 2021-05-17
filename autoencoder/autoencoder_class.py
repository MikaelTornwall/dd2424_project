"""
Code inspired by: https://debuggercafe.com/implementing-deep-autoencoder-in-pytorch/
"""

import os
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # encoder
        self.enc1 = nn.Linear(in_features=50, out_features=140)
        self.enc2 = nn.Linear(in_features=140, out_features=40)
        self.enc3 = nn.Linear(in_features=40, out_features=30)
        self.enc4 = nn.Linear(in_features=30, out_features=10)
        # decoder 
        self.dec1 = nn.Linear(in_features=10, out_features=30)
        self.dec2 = nn.Linear(in_features=30, out_features=40)
        self.dec3 = nn.Linear(in_features=40, out_features=140)
        self.dec4 = nn.Linear(in_features=140, out_features=50)

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        return x
