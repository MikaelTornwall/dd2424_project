"""
Training implementation based off: https://heartbeat.fritz.ai/guide-to-restricted-boltzmann-machines-using-pytorch-ee50d1ed21a8
"""

import pandas as pd
import numpy as np
import torch
from pre_train_rbm import RBM


# Procedure:
# 1. fetch the data
BC3_PICKLE_LOC  = "./data/dataframes/BC3_df_with_sentence_vectors.pkl"
BC3_df = pd.read_pickle(BC3_PICKLE_LOC)

# 2. Fetch the sentence vectors from the model
training_documents = BC3_df['df_vectors'].tolist()

for doc_vecs in training_documents:
    vecs = torch.FloatTensor(doc_vecs)
    # print('training vectors: ', doc_vecs)

# test with the first document! 
vecs = torch.FloatTensor(training_documents[0])
print(vecs)

# 3. set up the RBM
n_visible = len(vecs[0]) # should be the number of features (10)
print('n_visible: ', n_visible)
n_hidden = 140
batch_size = 2
n_sents = vecs.shape[0]
print('n_sents: ', n_sents)
rbm = RBM(n_visible, n_hidden)

W_pre, a_pre, b_pre = rbm.get_model_parameters()
print('W pre: ', W_pre)

# 3. train the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_sent in range(0, n_sents - batch_size, batch_size):
        vk = vecs[id_sent:id_sent+batch_size]
        v0 = vecs[id_sent:id_sent+batch_size]
        ph0,_ = rbm.sample_h(v0)
        # print('ph0: ', ph0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            # print('hk: ', hk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

W_pos, a_pos, b_pos = rbm.get_model_parameters()
print('W pos: ', W_pos)
# 4. pass the hidden units of the first RBM as the inputs to the second layer