from __future__ import unicode_literals, print_function, division
from functools import cmp_to_key
from io import open
from language import read_languages, prepare_data
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


"""
    Inspiration 
    from
    https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    and 
    https://www.kaggle.com/rahuldshetty/text-summarization-in-pytorch/comments
"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = pd.read_pickle(r'../data/dataframes/wrangled_BC3_df.pkl')
print(data.info())

X = data['body']
Y = data['summary']


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
    

clean_body = clean_text(X)
clean_summary = clean_text(Y)

print(data['tokenized_body'][0])
input_language, output_language, pairs = prepare_data(clean_body, clean_summary)


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
        return torch.zeros(1, 1, self.hidden_size, device=device)


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
        return torch.zeros(1, 1, self.hidden_size, device=device)

