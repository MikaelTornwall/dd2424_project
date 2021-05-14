"""
Training implementation based off: https://heartbeat.fritz.ai/guide-to-restricted-boltzmann-machines-using-pytorch-ee50d1ed21a8
"""

import pandas as pd
import numpy as np
import torch
from pre_train_rbm import RBM


# Procedure:
def train_rbm(vecs, n_hidden, input_size):
    # 3. set up the RBM
    n_visible = len(vecs[0]) # should be the number of features in the input layer
    print('n_visible: ', n_visible)
    batch_size = 100
    n_sents = vecs.shape[0]
    print('n_sents: ', n_sents)
    rbm = RBM(n_visible, n_hidden)

    W_pre, a_pre, b_pre = rbm.get_model_parameters()
    # print('W pre: ', W_pre)
    
    # save the hidden layer of the iteration with the minimal energy
    min_train_loss = 100
    hidden_layer_values = None

    # 3. train the RBM
    nb_epoch = 10
    for epoch in range(1, nb_epoch + 1):
        train_loss = 0
        s = 0.
        curr_hidden_layer = None
        for id_sent in range(0, n_sents - batch_size + 1, batch_size):
        # for id_sent in range(0, 1, batch_size):
            # print('in the for loop!')
            vk = vecs[id_sent:id_sent+batch_size]
            v0 = vecs[id_sent:id_sent+batch_size]
            ph0,_ = rbm.sample_h(v0)
            for k in range(10):
                _,hk = rbm.sample_h(vk)
                _,vk = rbm.sample_v(hk)
                vk[v0<0] = v0[v0<0]
            phk,vhk = rbm.sample_h(vk)
            rbm.train(v0, vk, ph0, phk)
            # this corresponds to the energy?
            train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
            s += 1.
            if curr_hidden_layer == None:
                curr_hidden_layer = vhk # TODO: what should we save as the vectors to the next layer???
            else:
                curr_hidden_layer = torch.cat((curr_hidden_layer, vhk), 0)
        loss = train_loss/s
        # set the hidden layer vectors to the vectors of the current layer
        if loss < min_train_loss:
            hidden_layer_values = curr_hidden_layer
            min_train_loss = loss
        print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

    W_pos, a_pos, b_pos = rbm.get_model_parameters()
    # print('W pos: ', W_pos)
    return W_pos, hidden_layer_values

def get_vector_data():
    # 1. fetch the data
    BC3_PICKLE_LOC  = "./data/dataframes/BC3_df_with_sentence_vectors.pkl"
    BC3_df = pd.read_pickle(BC3_PICKLE_LOC)

    # 2. Fetch the sentence vectors from the model
    training_documents = BC3_df['df_vectors'].tolist()

    # 3. Create one datastructure holding all the
    vecs = None
    for doc_vecs in training_documents:
        if len(doc_vecs): # don't include the empty lists
            v = torch.FloatTensor(doc_vecs)
            if vecs == None:
                vecs = v
            else:
                vecs = torch.cat((vecs, v), 0)
    print(vecs.shape)
    return vecs


def train_model_parameters():
    layers = [140, 40, 30, 10, 30, 40, 140, 10] # the last layer corresponds to the number of features in x.
    # the visible layer should first be the sentence vectors, and then the hidden layer of the previous RBM
    visible_layer = get_vector_data()
    all_model_params = []
    for layer in layers:
        input_size = visible_layer.shape[0]
        W, h_layer_vectors = train_rbm(visible_layer, layer, input_size) # input size is the number of sentences
        all_model_params.append(W)
        print('W shape: ', W.shape)
        # print('W: ', W)
        print('h_layer_vectors shape: ', h_layer_vectors.shape)
        # print('h_layer_vectors: ', h_layer_vectors)
        visible_layer = h_layer_vectors
    # TODO: put the weights 
    return all_model_params

all_model_params = train_model_parameters()