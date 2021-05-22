import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 10000
SOS_TOKEN = 0
EOS_TOKEN = 1

"""
    Hyperparameters
"""
TFR = 0.5
n_layers = 1
dropout = 0.1