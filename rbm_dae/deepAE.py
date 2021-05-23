
import pandas as pd
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from CGDs import ACGD
import matplotlib.pyplot as plt
from rbm_dae.DAE import DAE
from rbm_dae.RBM import RBM

"""

Pre-training phase with stacked RBM:s with layers 140,40,30,10
Fine-tuning phase back propagating with AE 

Based on implementation from article:
https://towardsdatascience.com/improving-autoencoder-performance-with-pretrained-rbms-e2e13113c782

With github page:
https://github.com/eugenet12/pytorch-rbm-autoencoder

"""
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

def stackedRBM(sentences, sentenceloader, train_batch_size):
    rbm_models = []
    visible_dim = sentences.shape[1]
    current_sentenceloader = sentenceloader

    for hidden_dim in [140, 40, 30, 10]:
        num_epochs = 15 # set to 15 so that we can get a fine-tune parameter search in the autoencoder
        learning_rate = 1e-3
        use_gaussian = hidden_dim == 10  # Use gaussian distribution on last layer


        rbm = RBM(visible_dim=visible_dim, hidden_dim=hidden_dim,
                  gaussian_hidden_distribution=use_gaussian)

        for epoch in range(num_epochs):
            for i, data_list in enumerate(current_sentenceloader):
                sample_data = data_list
                v0 = sample_data

                # Contrastive Divergence, no loop needed since k = 1 like original paper
                _, hk = rbm.sample_h(v0)
                pvk = rbm.sample_v(hk)

                # Not sure if we should keep momentum coefficient and weigh_decay?
                rbm.update_weights(v0, pvk, rbm.sample_h(v0)[0], rbm.sample_h(pvk)[0], learning_rate,
                                   momentum_coef=0.5 if epoch < 5 else 0.9,
                                   weight_decay=2e-4,
                                   batch_size=sample_data.shape[0])

        rbm_models.append(rbm)

        current_sentences = [rbm.sample_h(data_list)[1].detach().numpy() for data_list in current_sentenceloader]
        current_sentences = torch.FloatTensor(current_sentences)
        # problem: we get activations for a batch, and then this data is aggregated for each of the batches. 
        # fix: reshape so that the first two dimensions are merged!
        current_sentences = torch.flatten(current_sentences, start_dim=0, end_dim=1)

        current_sentenceloader = DataLoader(
            current_sentences,
            batch_size=train_batch_size,
            shuffle=False,
            drop_last=True
        )

        visible_dim = hidden_dim

    return rbm_models


def train_DAE(sentence_vectors, batch_size):
    train_loss = []
    train_batch_size = batch_size
    # update so that there will be at least one batch for small vector sets..
    if sentence_vectors.shape[0] < batch_size:
       train_batch_size = sentence_vectors.shape[0]

    sentenceloader = DataLoader(
        sentence_vectors,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True
    )

    # pre-training
    rbm_models = stackedRBM(sentence_vectors, sentenceloader, train_batch_size)

    # fine-tune Deep Auto Encoder
    dae = DAE(rbm_models)  # Using the pre-training RBM models as input, add False as second parameter for Naive AE

    # optimizer = optim.SGD(dae.parameters(), 1e-3)
    optimizer = optim.Adam(dae.parameters(), 1e-3)  # dae.parameters() includes all the parameters for the dae list(dae.parameters())

    # TODO: use MSE for glove vectors and BCE for df vectors!
    # To run BCE loss on glove vectors, need to add:
    # m = nn.Sigmoid()
    # batch_loss = criterion(m(dae(sentence)),sentence)

    # criterion = nn.MSELoss()                     # mean square loss function
    criterion = nn.BCELoss()                      # binary cross entropy loss function
    # The paper uses cross-entropy error as the loss function and
    # mini-batch CG with line search and the Polak-Ribiere rule for search direction.


    for epoch in range(100):
        running_loss = 0.0

        for i, sentence in enumerate(sentenceloader):

            # difference between actual sentence and reconstructed sentence
            batch_loss = criterion(dae(sentence), sentence)

            running_loss += batch_loss.item()                # dae(sentence) uses the forward method

            optimizer.zero_grad()   # We need to set the gradients to zero before starting to do backpropragation
            batch_loss.backward()   # Backward is the function which actually calculates the gradient which is stored internally in the tensors
            optimizer.step()        # step() makes the optimizer iterate over all parameters (tensors) it is supposed to update
                                    # and use their internally stored grad to update their values

        loss = running_loss / len(sentenceloader)
        train_loss.append(loss)
    return dae, train_loss

def train_autoencoder(df, vector_set):
    """
    Parameters: 
        df: the dataframe of the data to train on
        vector_set: the vector column to use. can be df_vectors or sentence_vectors (glove)
    """
    sentence_vectors = get_vector_data(df, vector_set)
    batch_size = 100

    dae, train_loss = train_DAE(sentence_vectors, batch_size)

    plt.figure()
    plt.plot(train_loss)
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    #plt.savefig('DAE__loss.png')

    return dae