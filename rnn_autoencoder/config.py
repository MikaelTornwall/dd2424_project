import torch

if torch.cuda.is_available():
    print('GPU available')
    DEVICE = torch.device('cuda')
    print('Using cuda')
else:
    print('GPU not available')
    DEVICE = torch.device('cpu')
    print('Using CPU')

MAX_LENGTH = 10000
SOS_TOKEN = 0
EOS_TOKEN = 1

"""
    Hyperparameters
"""
TFR = 0.5