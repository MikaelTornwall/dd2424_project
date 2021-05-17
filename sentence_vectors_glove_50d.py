"""
 Data parsing. Mainly borrowed from previous project using the EC3 dataset. 
Source code at: https://github.com/dailykirt/ML_Enron_email_summary/blob/master/notebooks/Text_rank_summarization.ipynb
"""

import pandas as pd
import numpy as np

# load the processed text, if function is not called fro sentence_vectors.py
BC3_PICKLE_LOC  = "./data/dataframes/wrangled_BC3_df.pkl"
BC3_df = pd.read_pickle(BC3_PICKLE_LOC)

'''This returns word vectors from the pretrained glove model. '''
def extract_word_vectors():
    word_embeddings = {}
    f = open('./data/wordvectors/glove.6B.50d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    return word_embeddings

'''
The tokenized sentences were done during preprocessing, 
so this function retrieves them from the dataframe, then flattens the list. 
'''
def get_tokenized_sentences(df):
    if (type(df.extractive_sentences) != list):
        clean_sentences = df.tokenized_body.tolist()
    else:
        clean_sentences = df.tokenized_body
    #flatten list
    clean_sentences = [y for x in clean_sentences for y in x]
    return clean_sentences

'''Create sentence_vectors for each tokenized sentence using the word_embeddings model. '''
def create_sentence_vectors(clean_sentences, word_embeddings):
    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((50,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((50,))
        sentence_vectors.append(v)
    return sentence_vectors

''' Looks through all the parsed data and creates sentence vectors for each data object
Adds a column with the sentence vectors to the data frame! '''
def create_sentence_vectors_for_all(word_embeddings, df):
    all_sentence_vectors = []
    for index, row in df.iterrows():
        data = pd.DataFrame([row])
        data['body'].iloc[0]
        clean_sentences = get_tokenized_sentences(data)
        sentence_vectors = create_sentence_vectors(clean_sentences, word_embeddings)
        all_sentence_vectors.append(sentence_vectors)

    df['sentence_vectors'] = all_sentence_vectors

    return df

"""
Update the pandas to include sentence vectors for each email object
Parameter: dataframe object
"""
def compute_and_add_glove_to_dataframe(df):
    word_embeddings = extract_word_vectors()
    BC3_df_with_vectors = create_sentence_vectors_for_all(word_embeddings, df)
    return BC3_df_with_vectors

''' Looks through all the parsed data and creates vectors for each SUMMARY
Adds a column with the summary vectors to the passed data frame! '''
# TODO: clean summary data if using them!!
def create_summary_vectors_for_all(word_embeddings, df):
    all_summary_vectors = []
    for index, row in df.iterrows():
        data = pd.DataFrame([row])
        data['subject'].iloc[0]
        clean_summaries = get_tokenized_sentences(data)
        summary_vectors = create_sentence_vectors(clean_summaries, word_embeddings)
        all_summary_vectors.append(summary_vectors)

    df['subject_vector'] = all_summary_vectors
    print(df['subject_vector'])
    print(df['subject'])
    return df

def add_glove_for_summary(df):
    word_embeddings = extract_word_vectors()
    BC3_with_sentence_vectors = create_summary_vectors_for_all(word_embeddings, df)
    return BC3_with_sentence_vectors
