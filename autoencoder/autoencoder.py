import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from autoencoder.autoencoder_class import Autoencoder, DAE
from autoencoder.rbm.pre_training import *
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader


def get_glove_vector_data(df, vector_set):

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

def train(net, trainloader, NUM_EPOCHS, criterion, optimizer):
    train_loss = []
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for sent in trainloader:
            optimizer.zero_grad()
            outputs = net(sent)
            loss = criterion(outputs, sent)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        loss = running_loss / len(trainloader)
        train_loss.append(loss)
        # print('Epoch {} of {}, Train Loss: {:.3f}'.format(epoch+1, NUM_EPOCHS, loss))
    return train_loss

# TODO: how to define the training and the test set?
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
BATCH_SIZE = 100

def train_autoencoder(df, vector_set):
    sentence_vectors = get_glove_vector_data(df, vector_set)
    trainloader = DataLoader(
        sentence_vectors, 
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    rbm_models = train_rbm_model_parameters(df, vector_set)
    net = DAE(rbm_models)
    # net = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # train the network
    print('training network..')
    train_loss = train(net, trainloader, NUM_EPOCHS, criterion, optimizer)
    
    # Include to plot the trianing loss! 
    plt.figure()
    plt.plot(train_loss)
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig('deep_ae_sent_loss.png')
    
    return net