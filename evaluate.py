from summarize import *
from autoencoder.autoencoder import *
import rouge

BC3_PICKLE_LOC  = "./data/dataframes/BC3_df_with_sentence_vectors.pkl"
    
BC3_df = pd.read_pickle(BC3_PICKLE_LOC)

evaluator = rouge.Rouge()
# create the autoencoder
net = train_autoencoder()

""" #There are three different human summaries for the same email. 
#Get a single email summary
masked_df = BC3_df[(BC3_df['to'] == 'w3c-wai-ig@w3.org')][:1]
listno = masked_df['listno'].iloc[0]
email_no = masked_df['email_num'].iloc[0]
masked_summaries = BC3_df[(BC3_df['listno'] == listno) & (BC3_df['email_num'] == email_no)]
masked_summaries.head()

#A summary that is an exact copy of the 'gold standard' should give a score of 1.0 
reference = masked_summaries['summary'].iloc[0]

ranked_sentences = summarize_on_sentence_vectors(masked_df)
print('ranked shape: ', ranked_sentences)
ranked_ae_sentences = summarize_autoencoder(masked_df, net)
print('ranked ae shape: ', ranked_ae_sentences)

# get the top ranked sentences
SUM_LEN = 2
sum = []
sum_ae = []
for i in range(SUM_LEN):
    sum.append(ranked_sentences[i][2])
    sum_ae.append(ranked_ae_sentences[i][2])

sum_str = ' '.join(sum)
sum_ae_str = ' '.join(sum_ae)

print('sum: ', sum_str)
print('sum ae: ', sum_ae_str)

sum_score = evaluator.get_scores(sum_str, reference)
print('sum score: ', sum_score)
sum_ae_score = evaluator.get_scores(sum_ae_str, reference)
print('sum ae score: ', sum_ae_score) """

# loop through all docs in the corpus
score_summary = []
score_summary_ae = []
for index, row in BC3_df.iterrows():
    df = pd.DataFrame([row])
    df['body'].iloc[0]
    reference = df['summary'].iloc[0]
    # print('reference: ', reference)
    print('iteration: ', index)
    ranked_sentences = summarize_on_sentence_vectors(df)
    # print('ranked shape: ', ranked_sentences)
    ranked_ae_sentences = summarize_autoencoder(df, net)
    # print('ranked ae shape: ', ranked_ae_sentences)
    SUM_LEN = 1
    if len(ranked_sentences) >= SUM_LEN:
        # get the top ranked sentences
        sum = []
        sum_ae = []
        for i in range(SUM_LEN):
            sum.append(ranked_sentences[i][2])
            sum_ae.append(ranked_ae_sentences[i][2])

        sum_str = ' '.join(sum)
        sum_ae_str = ' '.join(sum_ae)

        # print('sum: ', sum_str)
        # print('sum ae: ', sum_ae_str)

        sum_score = evaluator.get_scores(sum_str, reference)
        # print('sum score: ', sum_score)
        score_summary.append(sum_score[0]['rouge-1']['f'])
        sum_ae_score = evaluator.get_scores(sum_ae_str, reference)
        # print('sum ae score: ', sum_ae_score)
        score_summary_ae.append(sum_ae_score[0]['rouge-1']['f'])

# TODO: print the graphs for the extracted sentences
plt.plot(score_summary, label = "summary")
plt.plot(score_summary_ae, label = "summary ae")
plt.xlabel('Sentence')
plt.ylabel('ROUGE score')
title = "Textrank for original vectors vs autonecoder vectors"
plt.title(title)
plt.legend()
plt.show()