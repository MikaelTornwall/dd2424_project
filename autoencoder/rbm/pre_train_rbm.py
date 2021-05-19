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
        # init model params
        self.W = torch.randn(n_visible, n_hidden)
        self.h_bias = torch.zeros(1, n_hidden)
        self.v_bias = torch.zeros(1, n_visible)

        # parameters for learning with momentum
        self.W_momentum = torch.zeros(n_visible, n_hidden)
        self.h_bias_momentum = torch.zeros(n_hidden)
        self.v_bias_momentum = torch.zeros(n_visible)

    # This corresponds to eqn 3 in the paper!
    def sample_h(self, x):
        """Get sample hidden values and activation probabilities"""
        wx = torch.mm(x, self.W)
        activation = wx + self.h_bias
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    # this corresponds to eqn 4 in the paper!
    # TODO: update this to be Gaussian-Bernoulli for the first layer (now its Bernoulli-Bernoulli)
    def sample_v(self, y):
        """Get visible activation probabilities"""
        wy = torch.mm(y, self.W.t())
        activation = wy + self.v_bias
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    """
    v0: input vector containing sentence vector
    vk: visible nodes obtained after k samplings
    ph0: the vector of probabilities ?
    phk: probability of the hidden nodes after k samplings
    corresponds to eqn 12 in the paper!
    """
    def update_weights(self, v0, vk, ph0, phk, lr, momentum_coef, weight_decay, batch_size):
        # update to learning with momentum
        """Learning step: update parameters"""
        self.W_momentum *= momentum_coef
        self.W_momentum += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)

        self.h_bias_momentum *= momentum_coef
        self.h_bias_momentum += torch.sum((ph0 - phk), 0)

        self.v_bias_momentum *= momentum_coef
        self.v_bias_momentum += torch.sum((v0 - vk), 0)

        self.W += lr*self.W_momentum/batch_size
        self.h_bias += lr*self.h_bias_momentum/batch_size
        self.v_bias += lr*self.v_bias_momentum/batch_size

        self.W -= self.W * weight_decay # L2 weight decay
    
    def get_model_parameters(self):
        return self.W, self.h_bias, self.v_bias