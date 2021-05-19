import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 10000
SOS_TOKEN = 0
EOS_TOKEN = 1

"""
    Hyperparameters that maybe should be tuned
"""
TFR = 0.5