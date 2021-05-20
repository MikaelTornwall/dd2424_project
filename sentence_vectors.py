"""
Running this program will:
1. parse the bc3 data
2. compute sentence vectors based on glove
3. compute the df vectors for each document
4. combine these in to one datastructure and write it to file
"""
import pandas as pd
import numpy as np
from create_sentence_vectors import *
from sentence_vectors_glove_50d import *

"""
Calls funcitons to create the sentence vectors.
Sentence vectors based on term frequency (to be used for the learning part)
Sentence vectors based on glove (to be used in when comparing sentence similiarity)
"""
def vectorize():
    # load the parsed data
    BC3_PICKLE_LOC  = "./data/dataframes/wrangled_BC3_df.pkl"
    BC3_df = pd.read_pickle(BC3_PICKLE_LOC)

    # add the glove vectors for sentences
    bc3_glove_sents = compute_and_add_glove_to_dataframe(BC3_df)
    # TODO: add back to add glove vectors for the summaries
    # bc3_glove_sums = add_glove_for_summary(bc3_glove_sents)

    # add the normalized term frequency vectors
    n_features = 50
    bc3_glove_tf = compute_and_add_tf_to_dataframe(bc3_glove_sents, n_features)
    # remove the rows that does not have df vector representation
    bc3_glove_tf = bc3_glove_tf[bc3_glove_tf.df_vectors.notnull()]
    print(bc3_glove_tf)

    # Store the pandas including the sentence vectors
    BC3_PICKLE_LOC  = "./data/dataframes/BC3_df_with_sentence_vectors.pkl"
    bc3_glove_tf.to_pickle(BC3_PICKLE_LOC)

vectorize()
