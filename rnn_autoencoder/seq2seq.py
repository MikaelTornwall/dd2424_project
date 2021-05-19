"""
    Inspiration 
    from
    https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""


from __future__ import unicode_literals, print_function, division
from functools import cmp_to_key
from io import open
from language import read_languages, prepare_data
import config
import unicodedata
import string
import re
import random
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.optim as optim
# from project.bc3_parser import remove_stopwords


def remove_stopwords(sen):
    """
        From bc3_parser

        TODO:
            - Modify project so that importing resources from other folders becomes easy, 
            i.e. __init__.py to every folder or something like that, check that out
            - Create a single parser file that allows different summarizer tools to utilize the resources they need
    """
    stop_words = stopwords.words('english')
    new_sentence = " ".join([i for i in sen if i not in stop_words])
    return new_sentence


def clean_text(sentences):
    """
        Cleans email texts (body/summary) in a similar fashion as tokenize_email in bc3_parser, but
        additionally removes email addresses and URLs
        
        TODO: 
            - possibly map I'm --> I am, you're --> you are, and so on

        Parameters
        ----------
        sentences : list
            a list of sentences before they have been cleaned
        
        Returns 
        ----------
        list
            a list of cleaned sentences
    """
    # remove emails
    clean_sentences = [re.sub('\S*@\S*\s?', '', s) for s in sentences]    
    # remove urls
    clean_sentences = [re.sub('http://\S+|https://\S+', '', s) for s in clean_sentences]    
    clean_sentences = [re.sub('[^a-zA-Z ]', '', s) for s in clean_sentences]            
    clean_sentences = [s.lower() for s in clean_sentences]        
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    return clean_sentences


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=config.DEVICE)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=config.DEVICE)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=config.MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        # Look up table for word vectors
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        # Multilayer gated recurrent unit 
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=config.DEVICE)
