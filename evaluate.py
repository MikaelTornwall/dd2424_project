from summarize import *
from autoencoder.autoencoder import *
from summarize import *
import rouge


"""
Using textrank to extract the highest scoring sentences in the vector data
"""
def summarize_sentence_vectors(df):
    print('summarizing sentence vectors..') 
    sentence_vectors = df['sentence_vectors'].tolist()
    sentence_vectors = np.array(sentence_vectors)
    #Create a list of ranked sentences. 
    ranked_sentences = summarize_emails(df, sentence_vectors[0])
    # display_summary(df, ranked_sentences)
    return ranked_sentences

"""
Using textrank to extract the highest scoring sentences in the vectors created by the autoencoder
"""
def summarize_autoencoder_vectors(df, net):
    print('summarizing autoencoder sentence vectors..') 
    sentence_vectors = df['sentence_vectors'].tolist()
    torch_vectors = torch.tensor(sentence_vectors[0], dtype=torch.float32)
    output_vectors = net(torch_vectors)
    #Create a list of ranked sentences. 
    ranked_sentences = summarize_emails(df, output_vectors, True)
    # display_summary(df, ranked_sentences)
    return ranked_sentences

def evaluate_rankings(df_train, df_test, target, sum_len):
    evaluator = rouge.Rouge() 
    #create and train the autoencoder (see autoencoder module)
    net = train_autoencoder(df_train)

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
        reference = df_c[target].iloc[0] # reference that we score against (could be summary or subject)!
        print('reference: ', reference)
        # get the ranked sentences for the original and the ae modified sentence vectors
        ranked_sentences = summarize_sentence_vectors(df_c)
        ranked_ae_sentences = summarize_autoencoder_vectors(df_c, net)

        # SUM_LEN = 2 # the number of sentences to include in the generated summary
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
def analyze_and_plot_rouge_scores(sum_scores, ae_sum_scores):
    avg_scores = np.mean(sum_scores, axis=1)
    avg_scores_ae = np.mean(ae_sum_scores, axis=1)
    print('avg rouge scores: ', avg_scores)
    print('avg rouge scores ae: ', avg_scores_ae)

    # print the graphs for the extracted sentences
    # print('arr shape: ',sum_scores[0].tolist())
    x = np.arange(len(sum_scores[0])).tolist()
    # print('x', x)
    plt.plot(x, sum_scores[0].tolist(), label = "summary, f")
    plt.plot(x, ae_sum_scores[0].tolist(), label = "summary ae, f")
    plt.xlabel('Sentence')
    plt.ylabel('ROUGE score')
    plt.title('Textrank for original vectors vs autonecoder vectors')
    plt.legend()
    plt.show()

def evaluate_bc3():
    BC3_PICKLE_LOC  = "./data/dataframes/BC3_df_with_sentence_vectors.pkl"  
    BC3_df = pd.read_pickle(BC3_PICKLE_LOC)
    
    # evaluate on 'summary' or 'subject'
    target = 'summary'
    summary_len = 4
    # target = 'subject'
    # summary_len = 1
    # TODO: split in to some type of training and testset
    sum_scores, ae_sum_scores = evaluate_rankings(BC3_df, BC3_df, target, summary_len)
    analyze_and_plot_rouge_scores(sum_scores, ae_sum_scores)

def evaluate_spotify():
    SPOTIFY_PICKLE_TRAIN_LOC  = "./data/dataframes/spotify_train_vectors.pkl"
    SPOTIFY_PICKLE_TEST_LOC  = "./data/dataframes/spotify_test_vectors.pkl" 
    df_train = pd.read_pickle(SPOTIFY_PICKLE_TRAIN_LOC)
    df_test = pd.read_pickle(SPOTIFY_PICKLE_TEST_LOC)
    
    # evaluate on 'summary' or 'subject'
    target = 'episode_desc'
    summary_len = 4
    # target = 'subject'
    # summary_len = 1
    sum_scores, ae_sum_scores = evaluate_rankings(df_train, df_test, target, summary_len)
    analyze_and_plot_rouge_scores(sum_scores, ae_sum_scores)

evaluate_spotify()
# evaluate_bc3()

