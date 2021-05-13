"""
RBM class to perform the pre-training:
Partially based off: https://heartbeat.fritz.ai/guide-to-restricted-boltzmann-machines-using-pytorch-ee50d1ed21a8
"""

import numpy as np
import pandas as pd
import torch

"""
Class that given the number of hidden and number of visible nodes in the model sets up RBM
visible units should correspond to the number of features in the input data! 
"""
class RBM():
    def __init__(self, n_visible, n_hidden):
        self.W = torch.randn(n_hidden, n_visible)
        self.a = torch.randn(1, n_hidden)
        self.b = torch.randn(1, n_visible)

    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    """
    v0: input vector containing sentence vector
    vk: visible nodes obtained after k samplings
    ph0: the vector of probabilities ?
    phk: probability of the hidden nodes after k samplings
    """
    def train(self, v0, vk, ph0, phk):
        t_1 = torch.mm(v0.t(), ph0)
        t_2 = torch.mm(vk.t(), phk)
        sum = t_1 - t_2
        self.W += sum.t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)

    def get_model_parameters(self):
        return self.W, self.a, self.b