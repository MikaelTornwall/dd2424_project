import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

"""

Pre-training phase with stacked RBM:s with layers 140,40,30,10
Fine-tuning phase back propagating with AE 

Based on implementation from article:
https://towardsdatascience.com/improving-autoencoder-performance-with-pretrained-rbms-e2e13113c782

With github page:
https://github.com/eugenet12/pytorch-rbm-autoencoder

"""

def get_vector_data():
    # 1. fetch the data
    BC3_PICKLE_LOC = "./data/spotify_test_vectors_10.pkl"
    BC3_df = pd.read_pickle(BC3_PICKLE_LOC)

    # 'sentence_vectors' for glove vectors, precalculated word vectors
    # 'df_vectors' for tf-idf  , vectors of mathematical word representations calculated with the term_frequency

    # 2. Fetch the sentence vectors from the model
    training_documents = BC3_df['df_vectors'].tolist()

    # 3. Create one datastructure holding all the word representation vectors
    vecs = None
    for doc_vecs in training_documents:
        if len(doc_vecs):  # don't include the empty lists
            v = torch.FloatTensor(doc_vecs)
            if vecs == None:
                vecs = v
            else:
                vecs = torch.cat((vecs, v), 0)

    return vecs


class RBM():
    def __init__(self, visible_dim, hidden_dim, gaussian_hidden_distribution=False):
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.gaussian_hidden_distribution = gaussian_hidden_distribution

        # intialize parameters
        self.W = torch.randn(visible_dim, hidden_dim) * 0.1
        self.h_bias = torch.zeros(hidden_dim)  # visible --> hidden
        self.v_bias = torch.zeros(visible_dim)  # hidden --> visible

        # parameters for learning with momentum
        self.W_momentum = torch.zeros(visible_dim, hidden_dim)
        self.h_bias_momentum = torch.zeros(hidden_dim)
        self.v_bias_momentum = torch.zeros(visible_dim)

    def sample_h(self, v):
        """Get sample hidden values and activation probabilities"""

        activation = torch.mm(v, self.W) + self.h_bias
        if self.gaussian_hidden_distribution:
            return activation, torch.normal(activation, torch.tensor([1]))
        else:
            p = torch.sigmoid(activation)
            return p, torch.bernoulli(p)

    def sample_v(self, h):
        """Get visible activation probabilities"""
        activation = torch.mm(h, self.W.t()) + self.v_bias
        p = torch.sigmoid(activation)
        return p

    def update_weights(self, v0, vk, ph0, phk, lr,
                       momentum_coef, weight_decay, batch_size):
        """Learning step: update parameters"""
        self.W_momentum *= momentum_coef
        self.W_momentum += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)

        self.h_bias_momentum *= momentum_coef
        self.h_bias_momentum += torch.sum((ph0 - phk), 0)

        self.v_bias_momentum *= momentum_coef
        self.v_bias_momentum += torch.sum((v0 - vk), 0)

        self.W += lr * self.W_momentum / batch_size
        self.h_bias += lr * self.h_bias_momentum / batch_size
        self.v_bias += lr * self.v_bias_momentum / batch_size

        self.W -= self.W * weight_decay  # L2 weight decay

def stackedRBM(sentences, sentenceloader):
    rbm_models = []
    visible_dim = sentences.shape[1]
    current_sentenceloader = sentenceloader

    for hidden_dim in [140, 40, 30, 10]:
        num_epochs = 30 if hidden_dim == 10 else 10  # Less epochs for last layer, maybe remove?
        learning_rate = 1e-3 if hidden_dim == 10 else 0.1  # Changing the learning rate for last layer , maybe remove?
        use_gaussian = hidden_dim == 10  # Use gaussian distribution on last layer

        # print("Layer: ", visible_dim)

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
                                   momentum_coef=0.5 if epoch < 5 else 0.9,         # Remove ?
                                   weight_decay=2e-4,
                                   batch_size=sample_data.shape[0])

        rbm_models.append(rbm)

        current_sentences = [rbm.sample_h(data_list)[1].detach().numpy() for data_list in current_sentenceloader]
        current_sentences = torch.FloatTensor(current_sentences)

        current_sentenceloader = DataLoader(
            current_sentences,
            batch_size=100,
            shuffle=False,
            drop_last=True
        )

        visible_dim = hidden_dim

    return rbm_models

class DAE(nn.Module):
    """A Deep Autoencoder that takes a list of RBMs as input"""

    def __init__(self, models,use_models = True):
        """Create a deep autoencoder based on a list of RBM models
        Parameters
        ----------
        models: list[RBM]
            a list of RBM models to use for autoencoding
        """
        super(DAE, self).__init__()

        # extract weights from each model
        encoders = []
        encoder_biases = []
        decoders = []
        decoder_biases = []
        for model in models:
            encoders.append(nn.Parameter(model.W.clone()))
            encoder_biases.append(nn.Parameter(model.h_bias.clone()))
            decoders.append(nn.Parameter(model.W.clone()))
            decoder_biases.append(nn.Parameter(model.v_bias.clone()))

        # build encoders and decoders
        self.encoders = nn.ParameterList(encoders)
        self.encoder_biases = nn.ParameterList(encoder_biases)
        self.decoders = nn.ParameterList(reversed(decoders))
        self.decoder_biases = nn.ParameterList(reversed(decoder_biases))

        if not use_models:
            for encoder in self.encoders:
                torch.nn.init.xavier_normal_(encoder, gain = 1.0)
            for encoder_bias in self.encoder_biases:
                torch.nn.init.zeros_(encoder_bias)
            for decoder in self.decoders:
                torch.nn.init.xavier_normal_(decoder, gain = 1.0)
            for decoder_bias in decoder_biases:
                torch.nn.init.zeros_(decoder_bias)


    def forward(self, v):
        """Forward step
        Parameters
        ----------
        v: Tensor
            input tensor
        Returns
        -------
        Tensor
            a reconstruction of v from the autoencoder
        """
        # encode
        p_h = self.encode(v)

        # decode
        p_v = self.decode(p_h)

        return p_v

    def encode(self, v):  # for visualization, encode without sigmoid
        """Encode input
        Parameters
        ----------
        v: Tensor
            visible input tensor
        Returns
        -------
        Tensor
            the activations of the last layer
        """
        p_v = v
        activation = v
        for i in range(len(self.encoders)):
            W = self.encoders[i]
            h_bias = self.encoder_biases[i]
            activation = torch.mm(p_v, W) + h_bias
            p_v = torch.sigmoid(activation)

        # for the last layer, we want to return the activation directly rather than the sigmoid
        return activation

    def decode(self, h):
        """Encode hidden layer
        Parameters
        ----------
        h: Tensor
            activations from last hidden layer
        Returns
        -------
        Tensor
            reconstruction of original input based on h
        """
        p_h = h
        for i in range(len(self.encoders)):
            W = self.decoders[i]
            v_bias = self.decoder_biases[i]
            activation = torch.mm(p_h, W.t()) + v_bias
            p_h = torch.sigmoid(activation)
        return p_h

def train_DAE(sentence_vectors):
    train_loss = []

    sentenceloader = DataLoader(
        sentence_vectors,
        batch_size=100,
        shuffle=True,
        drop_last=True
    )

    # pre-training
    rbm_models = stackedRBM(sentence_vectors, sentenceloader)


    # fine-tune Deep Auto Encoder
    dae = DAE(rbm_models)  # Using the pre-training RBM models as input
    optimizer = optim.Adam(dae.parameters(), 1e-3)
    criterion = nn.MSELoss()                        # gradient descent ?

    # The paper uses cross-entropy error as the loss function and
    # mini-batch CG with line search and the Polak-Ribiere rule for search direction.

    # We could discuss if we think we have time to implement this
    # right now we are using Mean-squared Error and Adam optimizer

    for epoch in range(100):
        running_loss = 0.0

        for i, sentence in enumerate(sentenceloader):
            batch_loss = criterion(sentence, dae(sentence))  # difference between actual sentence and reconstructed sentence
            running_loss += batch_loss.item()

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        loss = running_loss / len(sentenceloader)
        train_loss.append(loss)
    return train_loss


sentence_vectors = get_vector_data()

train_loss = train_DAE(sentence_vectors)

# plt.figure()
# plt.plot(train_loss)
# plt.title('Train Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.savefig('glove_DAE__loss.png')
