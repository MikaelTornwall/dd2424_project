from os import read
import numpy as np
import torch
import config

SOS_token = 0
EOS_token = 1

class Language:
    """
        Class for representing the language used in our model, i.e. all the words that exist
        in the data (body / summary). The class helps us to represent each word in the language as a one-hot vector.
        
        Attributes
        ----------
        name        = name of the language (in our case this can be body / summary)
        word2index  = object that contains each word as the key and its index as the value
        word2count  = the number a word appears in the language
        index2word  = object that contains each index as the key and the corresponding word as the value
        n_word      = the number of unique words in the language

        Methods
        ----------
        add_sentence    = allows us to process complete sentences and add them into the language representation
        add_word        = adds a single word into the language representation

    """

    def __init__(self, summary):
        self.summary = summary
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: 'SOS', 1: 'EOS'}
        self.n_words = 2

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def read_languages(body, summary):
    """
    """
    print('Reading lines')
    pairs = [[body[i], summary[i]] for i in range(len(body))]
        
    input_language = Language(False)
    output_language = Language(True)

    return input_language, output_language, pairs


def prepare_data(language_1, language_2):
    """
    """
    input_language, output_language, pairs = read_languages(language_1, language_2)

    print(f'Reading {len(pairs)} sentence pairs...')
    print('Counting words')
    for pair in pairs:
        input_language.add_sentence(pair[0])
        output_language.add_sentence(pair[1])
    print('Counted words:')
    print(input_language.summary, input_language.n_words)
    print(output_language.summary, output_language.n_words)
    return input_language, output_language, pairs


def indices_from_text(language, text):
    """
        Function takes language and a text in that language and returns a list of the indices for each word in the given language

        Parameters
        ----------
        class : language
            Instance of the class Language that contains the complete vocabulary either for the text body or the summary
        string : text
            A single text document, either text body or summary

        Returns
        ----------
        list
            Returning list contains index of each word in the input text as it appears in the language
    """
    return [language.word2index[word] for word in text.split(' ')]

def tensor_from_text(language, text):
    """
       Function uses function indices_from_text to generate a list of indices of each corresponding word in a text in a language
       Then an End-Of-Sentence (EOS) token in appended and the list is translated into a tensor

        Parameters
        ----------
        class : language
            Instance of the class Language that contains the complete vocabulary either for the text body or the summary
        string : text
            A single text document, either text body or summary
        
        Returns
        ----------
        tensor
            Tensor containing the the list of indices and EOS token
    """
    indices = indices_from_text(language, text)
    indices.append(EOS_token)
    return torch.tensor(indices, dtype=torch.long, device=config.DEVICE).view(-1, 1)
    

def tensor_from_pair(input_language, target_language, pair):
    """
        Function takes a pair of text body and its summary and generates index tensors for each by utilizing tensor_from_text function

        Parameters
        ----------
        class : language
            Class intance of the text body

        class : language
            Class intance of the summary

        tuple : pair
            A tuple than contains text body in index 0 and its summary in index 1
        
        Returns
        ----------
        tuple
            Returns a tuple of tensors containing the word indices for the given pair of text body and summary
    """
    input_tensor = tensor_from_text(input_language, pair[0])
    target_tensor = tensor_from_text(target_language, pair[1])
    return (input_tensor, target_tensor)