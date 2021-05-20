from pandas.core.base import SelectionMixin
from summarize import *
from autoencoder.autoencoder import *
from summarize import *
import rouge



def summarize_sentence_vectors(df, vector_set):
    """
    Function applying the summarization function to get the ranked sentences. 
    Parameters:
        df: dataframe containing the data to summarize
        vector_set: the column name of the vector set to rank on
    Returns the ranked sentences
    """
    print('summarizing sentence vectors..')
    sentence_vectors = df[vector_set].tolist()
    sentence_vectors = np.array(sentence_vectors)
    #Create a list of ranked sentences. 
    ranked_sentences = summarize_emails(df, sentence_vectors[0])
    # display_summary(df, ranked_sentences)
    return ranked_sentences


def summarize_autoencoder_vectors(df, net, vector_set):
    """
    Function applying the autoencoder to the df vectors and the applying the summarization function to get the ranked sentences. 
    Parameters:
        df: dataframe containing the data to summarize
        net: trained autoencoder
        vector_set: the column name of the vector set to rank on
    Returns the ranked sentences
    """
    print('summarizing autoencoder sentence vectors..') 
    sentence_vectors = df[vector_set].tolist()
    torch_vectors = torch.tensor(sentence_vectors[0], dtype=torch.float32)
    output_vectors = net(torch_vectors)
    #Create a list of ranked sentences. 
    ranked_sentences = summarize_emails(df, output_vectors, True)
    # display_summary(df, ranked_sentences)
    return ranked_sentences

def evaluate_rankings(df_train, df_test, target, sum_len, corpus_ae=True, vector_set='sentence_vectors'):
    """
    Funtion to evaluate the returned summaries. the summaries are created baased on the raw sentence vectors and the autoencoder vectors
    Parameters:
        df_train: dataframe with the training data
        df_test: dataframe with the test data
        target: string containing the column that should be used as the summary reference
        sum_len: the number of sentences to include in the summary
        corpus_as: Boolean deciding wether to train the autoencoder on the entire corpus or on each document
        vector_set: column name of the column with the sentence vectors (can be glove vectors or tf vectors)
    Returns: the scores for the rouge parameters (3D matrix)
    """
    evaluator = rouge.Rouge() 
    #create and train the autoencoder (see autoencoder module)
    net = None
    if corpus_ae:
        net = train_autoencoder(df_train, vector_set)

    # loop through all docs in the corpus
    print('evaluating summaries..')
    df_len = int(df_test.shape[0])
    sum_scores = np.zeros((3, df_len))
    ae_sum_scores = np.zeros((3, df_len))
    curr_row = 0
    for index, row in df_test.iterrows():
        print('iteration: ', index)
        df_c = pd.DataFrame([row])
        df_c['body'].iloc[0]
        # Only proceed if the vectors of the current row are of correct dimensions (not [])
        if len(df_c[vector_set].tolist()[0]) > 0:
            # train AE on the current document only
            if not corpus_ae :
                net = train_autoencoder(df_c, vector_set)
            reference = df_c[target].iloc[0] # reference that we score against (could be summary or subject)!
            print('reference: ', reference)
            # get the ranked sentences for the original and the ae modified sentence vectors
            ranked_sentences = summarize_sentence_vectors(df_c, vector_set)
            ranked_ae_sentences = summarize_autoencoder_vectors(df_c, net, vector_set)

            if len(ranked_sentences) >= sum_len:
                # get the top ranked sentences
                sum = []
                sum_ae = []
                for i in range(sum_len):
                    sum.append(ranked_sentences[i][2])
                    sum_ae.append(ranked_ae_sentences[i][2])

                sum_str = ' '.join(sum)
                sum_ae_str = ' '.join(sum_ae)
                print('summary: ', sum_str)
                print('ae summary: ', sum_ae_str)

                # get the ROUGE scores for the ranked sentences and add to plot data
                sum_score = evaluator.get_scores(sum_str, reference)
                sum_ae_score = evaluator.get_scores(sum_ae_str, reference)
                sum_scores[0, curr_row] = sum_score[0]['rouge-1']['f']
                sum_scores[1, curr_row] = sum_score[0]['rouge-1']['p']
                sum_scores[2, curr_row] = sum_score[0]['rouge-1']['r']

                ae_sum_scores[0, curr_row] = sum_ae_score[0]['rouge-1']['f']
                ae_sum_scores[1, curr_row] = sum_ae_score[0]['rouge-1']['p']
                ae_sum_scores[2, curr_row] = sum_ae_score[0]['rouge-1']['r']
                curr_row += 1
    
    sum_scores = sum_scores[:, 0:curr_row]
    ae_sum_scores = ae_sum_scores[:, 0:curr_row]
    return sum_scores, ae_sum_scores
        

# calculating averages
def analyze_and_plot_rouge_scores(sum_scores, ae_sum_scores, metric, dataset_name):
    avg_scores = np.mean(sum_scores)
    avg_scores_ae = np.mean(ae_sum_scores)
    print('avg rouge scores: ', avg_scores)
    print('avg rouge scores ae: ', avg_scores_ae)

    # print the graphs for the extracted sentences
    x = np.arange(len(sum_scores)).tolist()
    label_1 = "Raw " + metric
    label_2 = "AE vector " + metric
    plt.plot(x, sum_scores.tolist(), label = label_1)
    plt.plot(x, ae_sum_scores.tolist(), label = label_2)
    plt.xlabel('Sentence')
    plt.ylabel('ROUGE score')
    title = "ROUGE " +metric + " for raw (mean: " + str(round(avg_scores, 3)) +") and AE (mean: "+str(round(avg_scores_ae, 3)) +") for " + dataset_name
    plt.title(title)
    plt.legend()
    plt.show()

def evaluate_bc3():
    """
    Base function to run and plot the ROUGE scores for the bc3 dataset
    """
    BC3_PICKLE_LOC  = "./data/dataframes/BC3_df_with_sentence_vectors.pkl"  
    BC3_df = pd.read_pickle(BC3_PICKLE_LOC)
    
    # evaluate on 'summary' or 'subject'
    target = 'summary'
    summary_len = 1
    # can set to use the df vectors ('df_vectors') or the glove vectors ('sentence_vectors')
    # if using df_vectors, the outoencoder should only look at each document, as the features cannot be transloated to other documents!
    corpus_ae = True
    vector_set = 'sentence_vectors'
    # TODO: split in to some type of training and testset
    sum_scores, ae_sum_scores = evaluate_rankings(BC3_df, BC3_df, target, summary_len, corpus_ae, vector_set)
    # plot F scores:
    analyze_and_plot_rouge_scores(sum_scores[0], ae_sum_scores[0], 'f-score', 'BC3 dataset')
    # plot precision
    analyze_and_plot_rouge_scores(sum_scores[1], ae_sum_scores[1], 'precision', 'BC3 dataset')
    # plot recall
    analyze_and_plot_rouge_scores(sum_scores[2], ae_sum_scores[2], 'recall', 'BC3 dataset')

def evaluate_spotify():
    """
    Base function to run and plot the ROUGE scores for the bc3 dataset
    """
    SPOTIFY_PICKLE_TRAIN_LOC  = "./data/dataframes/spotify_train_vectors.pkl"
    SPOTIFY_PICKLE_TEST_LOC  = "./data/dataframes/spotify_test_vectors.pkl" 
    df_train = pd.read_pickle(SPOTIFY_PICKLE_TRAIN_LOC)
    df_test = pd.read_pickle(SPOTIFY_PICKLE_TEST_LOC)
    
    # evaluate on 'summary' or 'subject'
    target = 'episode_desc'
    summary_len = 1
    corpus_ae = True # if false, the autoencoder is only trained on the sentences in the current document
    # can set to use the df vectors ('df_vectors') or the glove vectors ('sentence_vectors')
    vector_set = 'sentence_vectors'

    sum_scores, ae_sum_scores = evaluate_rankings(df_train, df_test, target, summary_len, corpus_ae, vector_set)
    # plot F scores:
    analyze_and_plot_rouge_scores(sum_scores[0], ae_sum_scores[0], 'f-score', 'Spotify dataset')
    # plot precision
    analyze_and_plot_rouge_scores(sum_scores[1], ae_sum_scores[1], 'precision', 'Spotify dataset')
    # plot recall
    analyze_and_plot_rouge_scores(sum_scores[2], ae_sum_scores[2], 'recall', 'Spotify dataset')

# evaluate_spotify()
evaluate_bc3()

