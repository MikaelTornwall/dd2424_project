"""
Training implementation based off: https://heartbeat.fritz.ai/guide-to-restricted-boltzmann-machines-using-pytorch-ee50d1ed21a8
"""

import pandas as pd
import numpy as np
import torch
from autoencoder.rbm.pre_train_rbm import RBM


def train_rbm(vecs, n_hidden, input_size):
    # 3. set up the RBM
    n_visible = len(vecs[0]) # should be the number of features in the input layer
    batch_size = 100
    n_sents = vecs.shape[0]
    if n_sents < batch_size:
        batch_size = n_sents
    lr = 1e-3
    rbm = RBM(n_visible, n_hidden)

    # 3. train the RBM
    nb_epoch = 10
    for epoch in range(1, nb_epoch + 1): # for the sepcified number of epochs 
        for id_sent in range(0, n_sents - batch_size + 1, batch_size): # for each possible batch. Updated!
            vk = vecs[id_sent:id_sent+batch_size]
            v0 = vecs[id_sent:id_sent+batch_size]
            ph0,_ = rbm.sample_h(v0)
            for k in range(10):
                _,hk = rbm.sample_h(vk)
                _,vk = rbm.sample_v(hk)
                vk[v0<0] = v0[v0<0]
            phk,vhk = rbm.sample_h(vk)
            # update weights (new implementation)
            rbm.update_weights(v0, vk, ph0, phk, lr, 
                                momentum_coef=0.5 if epoch < 5 else 0.9, 
                               weight_decay=2e-4, 
                               batch_size=100)

    return rbm # returning the whole model!

def get_vector_data(df, vector_set):

    # 2. Fetch the glove sentence vectors from the model
    training_documents = df[vector_set].tolist()

    # 3. Create one datastructure holding all the sentence vectors
    vecs = None
    for doc_vecs in training_documents:
        if len(doc_vecs): # don't include the empty lists
            v = torch.FloatTensor(doc_vecs)
            if vecs == None:
                vecs = v
            else:
                vecs = torch.cat((vecs, v), 0)
    return vecs

def train_rbm_model_parameters(df, vector_set):
    print('finding model parameters with RBM..')
    layers = [140, 40, 30, 10] # hidden layers to train
    # the visible layer should first be the sentence vectors, and then the hidden layer of the previous RBM
    visible_layer = get_vector_data(df, vector_set)
    all_models = []
    for hidden_layer in layers:
        input_size = visible_layer.shape[0]
        rbm = train_rbm(visible_layer, hidden_layer, input_size) # input size is the number of sentences
        all_models.append(rbm)
        visible_layer = rbm.sample_h(visible_layer)[0]
        # print ('W for layer: ', rbm.get_model_parameters()[0])
    return all_models

# models = train_rbm_model_parameters()