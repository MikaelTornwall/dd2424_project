from sentence_vectors_glove_50d import *
import pandas as pd
import numpy as np

"""
Calls functions to create sentence vectors for the provided dataframe. adds column to the existing dataframe.
Sentence vectors based on glove 
"""
def vectorize():
    # load the parsed data
    SPOTIFY_PICKLE_TRAIN_LOC  = "./data/dataframes/spotify_df_train.pkl"
    SPOTIFY_PICKLE_TEST_LOC  = "./data/dataframes/spotify_df_test.pkl"
    spotify_df_train = pd.read_pickle(SPOTIFY_PICKLE_TRAIN_LOC)
    spotify_df_test = pd.read_pickle(SPOTIFY_PICKLE_TEST_LOC)

    # add the glove vectors for sentences
    spotify_glove_sents_train = compute_and_add_glove_to_dataframe(spotify_df_train)
    spotify_glove_sents_test = compute_and_add_glove_to_dataframe(spotify_df_test)


    # Store the pandas including the sentence vectors
    SPOTIFY_PICKLE_TRAIN_VEC_LOC  = "./data/dataframes/spotify_train_vectors.pkl"
    SPOTIFY_PICKLE_TEST_VEC_LOC  = "./data/dataframes/spotify_test_vectors.pkl"
    spotify_glove_sents_train.to_pickle(SPOTIFY_PICKLE_TRAIN_VEC_LOC)
    spotify_glove_sents_test.to_pickle(SPOTIFY_PICKLE_TEST_VEC_LOC)

vectorize()