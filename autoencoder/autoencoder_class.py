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

class DAE(nn.Module):
    def __init__(self, models, use_models=True):
        """Create a deep autoencoder based on a list of RBM models"""
        super(DAE, self).__init__()

        # build encoders and decoders based on weights from each 
        self.encoders = nn.ParameterList([nn.Parameter(model.W.clone()) for model in models])
        self.encoder_biases = nn.ParameterList([nn.Parameter(model.h_bias.clone()) for model in models])
        self.decoders = nn.ParameterList([nn.Parameter(model.W.clone()) for model in reversed(models)])
        self.decoder_biases = nn.ParameterList([nn.Parameter(model.v_bias.clone()) for model in reversed(models)])
        
        if not use_models:
            for encoder in self.encoders:
                torch.nn.init.xavier_normal_(encoder, gain=1.0)
            for encoder_bias in self.encoder_biases:
                torch.nn.init.zeros_(encoder_bias)
            for decoder in self.decoders:
                torch.nn.init.xavier_normal_(decoder, gain=1.0)
            for decoder_bias in self.encoder_biases:
                torch.nn.init.zeros_(decoder_bias)

    def forward(self, v):
        """Forward step"""
        p_h = self.encode(v)
        return self.decode(p_h)

    def encode(self, v):
        """Encode input"""
        p_v = v
        for i in range(len(self.encoders)):
            activation = torch.mm(p_v, self.encoders[i]) + self.encoder_biases[i]
            p_v = torch.sigmoid(activation)
        # for the last layer, we want to return the activation directly rather than the sigmoid
        return activation

    def decode(self, h):
        """Decode hidden layer"""
        p_h = h
        for i in range(len(self.encoders)):
            activation = torch.mm(p_h, self.decoders[i].t()) + self.decoder_biases[i]
            p_h = torch.sigmoid(activation)
        return p_h