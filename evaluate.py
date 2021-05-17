from summarize import *
from autoencoder.autoencoder import *
from summarize import *
import rouge


"""
Using textrank to extract the highest scoring sentences in the vector data
"""
def summarize_sentence_vectors(df): 
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
    sentence_vectors = df['sentence_vectors'].tolist()
    torch_vectors = torch.tensor(sentence_vectors[0], dtype=torch.float32)
    output_vectors = net(torch_vectors)
    #Create a list of ranked sentences. 
    ranked_sentences = summarize_emails(df, output_vectors, True)
    # display_summary(df, ranked_sentences)
    return ranked_sentences

def evaluate_rankings(target, sum_len):
    BC3_PICKLE_LOC  = "./data/dataframes/BC3_df_with_sentence_vectors.pkl"  
    BC3_df = pd.read_pickle(BC3_PICKLE_LOC)

    evaluator = rouge.Rouge() 
    #create and train the autoencoder (see autoencoder module)
    net = train_autoencoder()

    # loop through all docs in the corpus
    print('evaluating summaries..')
    df_len = int(BC3_df.shape[0])
    sum_scores = np.zeros((3, df_len))
    ae_sum_scores = np.zeros((3, df_len))
    curr_row = 0
    for index, row in BC3_df.iterrows():
        df = pd.DataFrame([row])
        df['body'].iloc[0]
        reference = df[target].iloc[0] # reference that we score against (could be summary or subject)!
        # get the ranked sentences for the original and the ae modified sentence vectors
        ranked_sentences = summarize_sentence_vectors(df)
        ranked_ae_sentences = summarize_autoencoder_vectors(df, net)

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
    plt.plot(sum_scores[0], label = "summary, f")
    plt.plot(ae_sum_scores[0], label = "summary ae, f")
    plt.xlabel('Sentence')
    plt.ylabel('ROUGE score')
    title = "Textrank for original vectors vs autonecoder vectors"
    plt.title(title)
    plt.legend()
    plt.show()

# evaluate on 'summary' or 'subject'
target = 'summary'
summary_len = 4
# target = 'subject'
# summary_len = 1
sum_scores, ae_sum_scores = evaluate_rankings(target, summary_len)
analyze_and_plot_rouge_scores(sum_scores, ae_sum_scores)

