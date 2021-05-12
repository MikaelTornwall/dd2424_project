import numpy as np
import pandas as pd
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer # used to extract term frequency for each document
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

BC3_PICKLE_LOC  = "./data/dataframes/wrangled_BC3_df.pkl"
# Find the n most common words in the document

"""
Function to determine what words are the most frequent across the document, and use those for the feature vector
Parameter: n. Number of most common words
Returns: Boolean stating wether the document can be used (has enough features)
Returns: array of words that are the most common in the document
"""
def find_common_words(doc, n):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(doc)
    X = X.toarray()
    X = np.array(X)
    # TODO: fix this in the parsing step??
    # check so that the document is long enough!
    if X.shape[0] < 1 or X.shape[1] < n:
        return False, []

    word_sums = X.sum(axis=0)
    indicies = np.argpartition(word_sums, -n)[-n:]
    names = vectorizer.get_feature_names()
    final_words = []
    for i in indicies:
        final_words.append(names[i])

    return True, final_words

"""
Function that computes the term frequency for each of the sentences in each of the documents
Parameter: The number of features to use in the vector representation
Returns: Dictionary with dictionary index as key and term frequency matrix as value
"""
def compute_tf_for_docs(n_features):
    # read and parse the data
    bc3_df = pd.read_pickle(BC3_PICKLE_LOC)
    docs = bc3_df['tokenized_body'].tolist()
    
    # data structure to hold the term frequencies for each document (the key is document row in the panda)
    doc_tfs = {}

    for idx, d in enumerate(docs):
        # Create the vocabulary for each document that is of common length across documents
        include, vocabulary = find_common_words(d, n_features)
        # if the document has enough words to be used (more than the wanted number of features)
        if include:
            vectorizer =  TfidfVectorizer(vocabulary=vocabulary, use_idf=False)
            X = vectorizer.fit_transform(d)
            X = np.array(X.toarray())

            # Adding small values to the feature vectors
            scores = np.random.normal(0, 1e-8, (X.shape))
            scores = scores + X
            doc_tfs[idx] = scores

    return doc_tfs

n_features = 10
tfs_for_docs = compute_tf_for_docs(n_features)