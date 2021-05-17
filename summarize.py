import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

def get_extractive_sentences(df):
    '''Retrieve original email sentences and index them. This will be used to generate the extracted summaries. '''
    if (type(df.extractive_sentences) != list):
        sentences_list = df.extractive_sentences.tolist()
    else:
        sentences_list = df.extractive_sentences
    #flatten list as tuples containting (sentence, dataframe index)  to reassociate summary with original email. 
    sentences = []
    for counter, sublist in enumerate(sentences_list):
        for item in sublist:
            sentences.append([counter, item]) 
    return sentences

def rank_sentences(sentences, sentence_vectors, are_tensors):
    '''This function takes in a list of sentences to input into TextRank. The resulting ranks are what the model 
    calculated as the most important sentences. '''
    sim_mat = np.zeros([len(sentences), len(sentences)])
    # Make the tensors (when using ae) to np arrays
    if are_tensors:
        sentence_vectors = sentence_vectors.detach().numpy()
    #Initialize matrix with cosine similarity scores. 
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                shape_len = len(sentence_vectors[i])
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,shape_len), sentence_vectors[j].reshape(1,shape_len))[0,0]
    nx_graph = nx.from_numpy_array(sim_mat)
    
    #Pair sentence with it's similarity score then sort.    
    try:
        scores = nx.pagerank(nx_graph)
        ranked_sentences = sorted(((scores[i],s[0],s[1]) for i,s in enumerate(sentences)), reverse=True)
    except:
        ranked_sentences = sorted(((0,s[0],s[1]) for i,s in enumerate(sentences)), reverse=True)

    return ranked_sentences

def summarize_emails(masked_df, sentence_vectors, are_tensors=False):
    '''Function to wrap up summarization process'''
    if len(masked_df) != 1:
        print("Total number of emails to summarize: " + str(len(masked_df)))
    # get the sentences from the current document
    sentences = get_extractive_sentences(masked_df)
    ranked_sentences = rank_sentences(sentences, sentence_vectors, are_tensors)
    return ranked_sentences

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def display_summary(enron_masked_df, ranked_sentences):
  '''Specify number of sentences as a fraction of total emails. '''
  sn = (len(enron_masked_df) // 10) + 1

  # Generate summary
  for i in range(sn):
    #pull date and subject from original email
    email_date = str(enron_masked_df['date'].iloc[ranked_sentences[i][1]])
    email_subject = str(enron_masked_df['subject'].iloc[ranked_sentences[i][1]])
    email_from = str(enron_masked_df['from'].iloc[ranked_sentences[i][1]])
    print( bcolors.BOLD + "Date: "+ email_date  + 
          " Subject: " + email_subject +
          " From: " + email_from + bcolors.ENDC +
          "\nSummary: " + str(ranked_sentences[i][2]))
