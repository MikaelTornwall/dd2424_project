from typing import Sequence
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from autoencoder.autoencoder_class import Autoencoder
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from summarize import *


def get_glove_vector_data():
    # 1. fetch the data
    BC3_PICKLE_LOC  = "./data/dataframes/BC3_df_with_sentence_vectors.pkl"
    BC3_df = pd.read_pickle(BC3_PICKLE_LOC)

    # 2. Fetch the glove sentence vectors from the model
    training_documents = BC3_df['sentence_vectors'].tolist()

    # 3. Create one datastructure holding all the sentence vectors
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

def get_glove_query_vector_data():
    # 1. fetch the data
    BC3_PICKLE_LOC  = "../data/dataframes/BC3_df_sentence_summary_vectors.pkl"
    BC3_df = pd.read_pickle(BC3_PICKLE_LOC)

    # column 'subject' or 'extractive_sentences'
    # 2. Fetch the subjects from the model
    summary_vectors = BC3_df['subject_vector'].tolist()
    print(BC3_df.columns)

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

def train_autoencoder():
    sentence_vectors = get_glove_vector_data()
    trainloader = DataLoader(
        sentence_vectors, 
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    net = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # train the network
    print('training network..')
    train_loss = train(net, trainloader, NUM_EPOCHS, criterion, optimizer)
    plt.figure()
    plt.plot(train_loss)
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('deep_ae_sent_loss.png')
    return net

# test the network! HOW??
# plan: get all the queries (email titles)
# run the autoendoder on the sentences in the doc and query
# get the cosine distance between and return the best matching sentences (based on the x_hat vectors)

# get_glove_query_vector_data()
def summarize_autoencoder(df, net):
    # print('retrieving autoencoder sentences..')
    sentence_vectors = df['sentence_vectors'].tolist()
    torch_vectors = torch.tensor(sentence_vectors[0], dtype=torch.float32)
    output_vectors = net(torch_vectors)
    #Create a list of ranked sentences. 
    ranked_sentences = summarize_emails(df, output_vectors, True)
    # display_summary(df, ranked_sentences)
    return ranked_sentences

# summarize_autoencoder()