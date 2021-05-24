from summarize import *
from rbm_dae.deepAE import *
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

def evaluate_rankings(df_train, df_test, target, sum_lens, corpus_ae=True, vector_set='sentence_vectors'):
    """
    Funtion to evaluate the returned summaries. the summaries are created baased on the raw sentence vectors and the autoencoder vectors
    Parameters:
        df_train: dataframe with the training data
        df_test: dataframe with the test data
        target: string containing the column that should be used as the summary reference
        sum_len: An array holding the number of sentences to include in the summary
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
    sum_scores = np.zeros((len(sum_lens), 3, 3, df_len))
    ae_sum_scores = np.zeros((len(sum_lens), 3, 3, df_len))
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
            # collecting the scores for the specified summary lengths
            for s_len in sum_lens:
                print('s_len: ', s_len)
                if len(ranked_sentences) >= s_len:
                    # get the top ranked sentences
                    sum = []
                    sum_ae = []
                    for i in range(s_len):
                        sum.append(ranked_sentences[i][2])
                        sum_ae.append(ranked_ae_sentences[i][2])

                    sum_str = ' '.join(sum)
                    sum_ae_str = ' '.join(sum_ae)
                    print('summary: ', sum_str)
                    print('ae summary: ', sum_ae_str)

                    # get the ROUGE scores for the ranked sentences and add to plot data
                    sum_score = evaluator.get_scores(sum_str, reference)
                    sum_ae_score = evaluator.get_scores(sum_ae_str, reference)
                    sum_scores[s_len-1, 0, 0, curr_row] = sum_score[0]['rouge-1']['f']
                    sum_scores[s_len-1, 0, 1, curr_row] = sum_score[0]['rouge-1']['p']
                    sum_scores[s_len-1, 0, 2, curr_row] = sum_score[0]['rouge-1']['r']
                    sum_scores[s_len-1, 1, 0, curr_row] = sum_score[0]['rouge-2']['f']
                    sum_scores[s_len-1, 1, 1, curr_row] = sum_score[0]['rouge-2']['p']
                    sum_scores[s_len-1, 1, 2, curr_row] = sum_score[0]['rouge-2']['r']
                    sum_scores[s_len-1, 2, 0, curr_row] = sum_score[0]['rouge-l']['f']
                    sum_scores[s_len-1, 2, 1, curr_row] = sum_score[0]['rouge-l']['p']
                    sum_scores[s_len-1, 2, 2, curr_row] = sum_score[0]['rouge-l']['r']

                    ae_sum_scores[s_len-1, 0, 0, curr_row] = sum_ae_score[0]['rouge-1']['f']
                    ae_sum_scores[s_len-1, 0, 1, curr_row] = sum_ae_score[0]['rouge-1']['p']
                    ae_sum_scores[s_len-1, 0, 2, curr_row] = sum_ae_score[0]['rouge-1']['r']
                    ae_sum_scores[s_len-1, 1, 0, curr_row] = sum_ae_score[0]['rouge-2']['f']
                    ae_sum_scores[s_len-1, 1, 1, curr_row] = sum_ae_score[0]['rouge-2']['p']
                    ae_sum_scores[s_len-1, 1, 2, curr_row] = sum_ae_score[0]['rouge-2']['r']
                    ae_sum_scores[s_len-1, 2, 0, curr_row] = sum_ae_score[0]['rouge-l']['f']
                    ae_sum_scores[s_len-1, 2, 1, curr_row] = sum_ae_score[0]['rouge-l']['p']
                    ae_sum_scores[s_len-1, 2, 2, curr_row] = sum_ae_score[0]['rouge-l']['r']
            curr_row += 1
    
    sum_scores = sum_scores[:, :, :, 0:curr_row]
    ae_sum_scores = ae_sum_scores[:, :, :, 0:curr_row]
    return sum_scores, ae_sum_scores
        

# calculating averages
def analyze_and_plot_rouge_scores(sum_scores, ae_sum_scores, metric, dataset_name, summary_len):
    avg_scores = np.mean(sum_scores)
    avg_scores_ae = np.mean(ae_sum_scores)
    print(dataset_name)
    print('Summary length: ', summary_len)
    raw_mean = 'Mean ' + metric + ' for raw vectors: ' + str(round(avg_scores, 3))
    dae_mean = 'Mean ' + metric + ' for DAE vectors: ' + str(round(avg_scores_ae, 3))
    print(raw_mean)
    print(dae_mean)

    # Add to plot graphs for the extracted sentences
    """ 
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
    """

def evaluate_bc3():
    """
    Base function to run and plot the ROUGE scores for the bc3 dataset
    """
    BC3_PICKLE_LOC  = "./final_data/BC3_127.pkl"  
    BC3_df = pd.read_pickle(BC3_PICKLE_LOC)
    # df contains 127 rows that all have df_vectors representation!
    # Split into training and test set
    BC3_df_train = BC3_df.iloc[:117]
    BC3_df_test = BC3_df.iloc[117:]
    
    # evaluate on 'summary' or 'subject'
    target = 'summary'
    summary_len = [1]
    # can set to use the df vectors ('df_vectors') or the glove vectors ('sentence_vectors')
    corpus_ae = True
    vector_set = 'sentence_vectors'                #df_vectors
    sum_scores, ae_sum_scores = evaluate_rankings(BC3_df_train, BC3_df_test, target, summary_len, corpus_ae, vector_set)
    plot_all_scores(sum_scores, ae_sum_scores, 'bc3 dataset', summary_len[0])

def evaluate_spotify():
    """
    Base function to run and plot the ROUGE scores for the spotify dataset
    """
    SPOTIFY_PICKLE_TRAIN_LOC  = "./final_data/spotify_train_422.pkl"
    SPOTIFY_PICKLE_TEST_LOC  = "./final_data/spotify_test_45.pkl" 
    df_train = pd.read_pickle(SPOTIFY_PICKLE_TRAIN_LOC)
    df_test = pd.read_pickle(SPOTIFY_PICKLE_TEST_LOC)

    # section to get the summary for a specidic episode
    # df_sent = df_train.loc[df_train['episode_id'] == '7DoDuJE4sCBu2jJlOgCrwA']
    # df_test = df_sent

    target = 'episode_desc'
    summary_len = [1]
    corpus_ae = True # if false, the autoencoder is only trained on the sentences in the current document
    # can set to use the df vectors (t-idf) ('df_vectors') or the glove vectors ('sentence_vectors')
    vector_set = 'sentence_vectors'


    sum_scores, ae_sum_scores = evaluate_rankings(df_train, df_test, target, summary_len, corpus_ae, vector_set)
    plot_all_scores(sum_scores, ae_sum_scores, 'spotify dataset', summary_len[0])

def plot_all_scores(sum_scores, ae_sum_scores, dataset, summary_len):
    """
    Base function to plot ROUGE scores.
    Parameters:
        - sum_scores: Matirx of scores for the raw vectors.
        - as_sum_scores: Matrix of scores for the vectors produced by autoencoder
    """
    analyze_and_plot_rouge_scores(sum_scores[0][0][0], ae_sum_scores[0][0][0], 'rouge-1 f-score', dataset, summary_len)
    analyze_and_plot_rouge_scores(sum_scores[0][0][1], ae_sum_scores[0][0][1], 'rouge-1 precision', dataset, summary_len)
    analyze_and_plot_rouge_scores(sum_scores[0][0][2], ae_sum_scores[0][0][2], 'rouge-1 recall', dataset, summary_len)

    # plot rouge-2 scores:
    analyze_and_plot_rouge_scores(sum_scores[0][1][0], ae_sum_scores[0][1][0], 'rouge-2 f-score', dataset, summary_len)
    analyze_and_plot_rouge_scores(sum_scores[0][1][1], ae_sum_scores[0][1][1], 'rouge-2 precision', dataset, summary_len)
    analyze_and_plot_rouge_scores(sum_scores[0][1][2], ae_sum_scores[0][1][2], 'rouge-2 recall', dataset, summary_len)

    # plot rouge-l scores:
    analyze_and_plot_rouge_scores(sum_scores[0][2][0], ae_sum_scores[0][2][0], 'rouge-l f-score', dataset, summary_len)
    analyze_and_plot_rouge_scores(sum_scores[0][2][1], ae_sum_scores[0][2][1], 'rouge-l precision', dataset, summary_len)
    analyze_and_plot_rouge_scores(sum_scores[0][2][2], ae_sum_scores[0][2][2], 'rouge-l recall', dataset, summary_len)


def get_mean(sum_scores, ae_sum_scores):
    """
    Function to get the mean of a vector of scores.
    """
    avg_scores = np.mean(sum_scores)
    avg_scores_ae = np.mean(ae_sum_scores)
    return avg_scores, avg_scores_ae

def evaluate_sentence_length_performance(df_train, df_test, target, summary_len, corpus_ae, vector_set, dataset):
    """
    Function to cumpute the rouge scores for a range of summary lengths.
    """
    averages_p = np.zeros((summary_len, 2))
    averages_ae_p = np.zeros((summary_len, 2))
    averages_r = np.zeros((summary_len, 2))
    averages_ae_r = np.zeros((summary_len, 2))
    summary_lengths = [1, 2, 3, 4, 5, 6]
    sum_scores, ae_sum_scores = evaluate_rankings(df_train, df_test, target, summary_lengths, corpus_ae, vector_set)
    for i in range(1, summary_len): 
        print('evaluating rankings for # sentences: ', i)
        for j in range(2): # for rouge-1 and rouge-2
                avg_score_p, avg_score_ae_p = get_mean(sum_scores[i-1][j][1], ae_sum_scores[i-1][j][1])
                avg_score_r, avg_score_ae_r = get_mean(sum_scores[i-1][j][2], ae_sum_scores[i-1][j][2])

                averages_p[i, j] = avg_score_p
                averages_ae_p[i, j] = avg_score_ae_p
                averages_r[i, j] = avg_score_r
                averages_ae_r[i, j] = avg_score_ae_r
    print('averages: ', averages_p)
    print('averages ae: ', averages_ae_p)
    averages_p = averages_p[1:].transpose()
    averages_ae_p = averages_ae_p[1:].transpose()
    averages_r = averages_r[1:].transpose()
    averages_ae_r = averages_ae_r[1:].transpose()
    return averages_p, averages_ae_p, averages_r, averages_ae_r

def plot_sentences(glove_averages, glove_averages_ae, df_averages, df_averages_ae, title, dataset):
    """
    Function to plot the mean scores vs sentence lengths for the different sentence encodings
    """
    x = np.arange(1,7).tolist()
    plt.plot(x, glove_averages.tolist(), label = "Glove vector")
    plt.plot(x, glove_averages_ae.tolist(), label = "Glove DAE vector")
    plt.plot(x, df_averages.tolist(), label = 'tf-idf vector')
    plt.plot(x, df_averages_ae.tolist(), label = 'tf-idf DAE vector')
    plt.xlabel('Number of sentences')
    plt.ylabel(title)
    t = title + ' for ' + dataset
    plt.title(t)
    plt.legend()
    plt.show()

def run_sentence_length_evaluation():
    """
    Main function to compute the mean scores for each summary length for the two datasets.
    """
    BC3_PICKLE_LOC  = "./final_data/BC3_127.pkl"  
    BC3_df = pd.read_pickle(BC3_PICKLE_LOC)
    # df contains 127 rows that all have df_vectors representation!
    # Split into training and test set
    bc3_df_train = BC3_df.iloc[:117]
    bc3_df_test = BC3_df.iloc[117:]
    bc3_target = 'summary'

    SPOTIFY_PICKLE_TRAIN_LOC  = "./final_data/spotify_train_422.pkl"
    SPOTIFY_PICKLE_TEST_LOC  = "./final_data/spotify_test_45.pkl" 
    s_df_train = pd.read_pickle(SPOTIFY_PICKLE_TRAIN_LOC)
    s_df_test = pd.read_pickle(SPOTIFY_PICKLE_TEST_LOC)
    s_target = 'episode_desc'

    summary_len = 7
    corpus_ae = True
    vector_set = 'sentence_vectors'
    df_vector_set = 'df_vectors'
    # metric = 0 # 0 = f-score, 1 = precision, 2 = recall
    bc3_glove_p, bc3_glove_ae_p, bc3_glove_r, bc3_glove_ae_r = evaluate_sentence_length_performance(bc3_df_train, bc3_df_test, bc3_target, summary_len, corpus_ae, vector_set, 'bc3 dataset')
    bc3_df_p, bc3_df_ae_p, bc3_df_r, bc3_df_ae_r = evaluate_sentence_length_performance(bc3_df_train, bc3_df_test, bc3_target, summary_len, corpus_ae, df_vector_set, 'bc3 dataset')
    plot_sentences(bc3_glove_p[0], bc3_glove_ae_p[0], bc3_df_p[0], bc3_df_ae_p[0], 'ROUGE-1 scores precision', 'BC3 dataset')
    plot_sentences(bc3_glove_p[1], bc3_glove_ae_p[1], bc3_df_p[1], bc3_df_ae_p[1], 'ROUGE-2 scores precision', 'BC3 dataset')
    plot_sentences(bc3_glove_r[0], bc3_glove_ae_r[0], bc3_df_r[0], bc3_df_ae_r[0], 'ROUGE-1 scores recall', 'BC3 dataset')
    plot_sentences(bc3_glove_r[1], bc3_glove_ae_r[1], bc3_df_r[1], bc3_df_ae_r[1], 'ROUGE-2 scores recall', 'BC3 dataset')

    s_glove_p, s_glove_ae_p, s_glove_r, s_glove_ae_r = evaluate_sentence_length_performance(s_df_train, s_df_test, s_target, summary_len, corpus_ae, vector_set, 'Spotify dataset')
    s_df_p, s_df_ae_p, s_df_r, s_df_ae_r = evaluate_sentence_length_performance(s_df_train, s_df_test, s_target, summary_len, corpus_ae, df_vector_set, 'Spotify dataset')
    plot_sentences(s_glove_p[0], s_glove_ae_p[0], s_df_p[0], s_df_ae_p[0], 'ROUGE-1 scores precision', 'Spotify dataset')
    plot_sentences(s_glove_p[1], s_glove_ae_p[1], s_df_p[1], s_df_ae_p[1], 'ROUGE-2 scores precision', 'Spotify dataset')
    plot_sentences(s_glove_r[0], s_glove_ae_r[0], s_df_r[0], s_df_ae_r[0], 'ROUGE-1 scores recall', 'Spotify dataset')
    plot_sentences(s_glove_r[1], s_glove_ae_r[1], s_df_r[1], s_df_ae_r[1], 'ROUGE-2 scores recall', 'Spotify dataset')

# evaluate_bc3()
evaluate_spotify()
# run_sentence_length_evaluation()