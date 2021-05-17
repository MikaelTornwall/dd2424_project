import json
import os
import re
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

"""
This function removes stopwords
"""
def remove_stopwords(sen):
    stop_words = stopwords.words('english')
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

def tokenize_email(text):
    clean_sentences = sent_tokenize(text, language='english')
    #removing punctuation, numbers and special characters. Then lowercasing. 
    clean_sentences = [re.sub('[^a-zA-Z ]', '',s) for s in clean_sentences]
    clean_sentences = [s.lower() for s in clean_sentences]
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    return clean_sentences

# TODO: read in the metadata. Separate out each episode into a dict
print('reading metadata..')
episode_dict = {}
with open('./data/spotify/outmeta.json', 'r') as j:
    json_data = json.load(j) 
    for name, data in json_data.items():
        pod_id = name.split(':')[2]
        podcast_name = data['podcast_name']
        podcast_desc = data['podcast_desc']
        for episode in data['episodes']:
            ep_id = episode['episode_filename_prefix']
            ep_desc = episode['episode_desc']
            ep_name = episode['episode_name']
            ep_obj = {
                'ep_name': ep_name,
                'ep_desc': ep_desc,
                'pod_id': pod_id,
                'pod_name': podcast_name,
                'pod_desc': podcast_desc,
            }
            episode_dict[ep_id] = ep_obj

# TODO: read all files. merge all the transcripts to one text block
# TODO: connect each text block to the description in the metadata
# TODO: create pd dataframe of columns 'body' 'description' 'episode_name' 'podcast_name'...
print('reading files..')
training_data = []
test_data = []
dir_name = './data/spotify/transformed/'   
# dir_name = "data"
files = [os.path.join(dir_name, fn) for fn in os.listdir(dir_name)]
# print(files)
num_files = 0
for file in files:
    # print(file)
    if num_files > 500: # only read 100 files for now..
        break
    num_files += 1
    with open(file, 'r') as j:
        json_data = json.load(j)
        id = json_data['id']
        if id in episode_dict:
            ep_metadata = episode_dict[id]
            # print(ep_metadata)
            body_arr = []
            for item in json_data['data']:
                if 'transcript' in item:
                    body_arr.append(item['transcript'])
            body = ' '.join(body_arr)
            episode_object = {
                'body': body,
                'episode_id': id,
                'episode_name': ep_metadata['ep_name'],
                'episode_desc': ep_metadata['ep_desc'],
                'pod_id': ep_metadata['pod_id'],
                'pod_name': ep_metadata['pod_name'],
                'pod_desc': ep_metadata['pod_desc'],
            }
            if num_files % 10 == 0:
                test_data.append(episode_object)
            else:
                training_data.append(episode_object)

df_spotify_train = pd.DataFrame(training_data)
df_spotify_test = pd.DataFrame(test_data)
print(df_spotify_train)
print(df_spotify_test)
print('cleaning data..')
# Tokenizing
df_spotify_train['extractive_sentences'] = df_spotify_train['body'].apply(sent_tokenize)
df_spotify_train['tokenized_body'] = df_spotify_train['body'].apply(tokenize_email)
df_spotify_test['extractive_sentences'] = df_spotify_test['body'].apply(sent_tokenize)
df_spotify_test['tokenized_body'] = df_spotify_test['body'].apply(tokenize_email) # function defined in the bc3 parser module!

""" 
Store data
"""
SPOTIFY_PICKLE_TRAIN_LOC  = "./data/dataframes/spotify_df_train.pkl"
SPOTIFY_PICKLE_TEST_LOC  = "./data/dataframes/spotify_df_test.pkl"
df_spotify_train.to_pickle(SPOTIFY_PICKLE_TRAIN_LOC)
df_spotify_test.to_pickle(SPOTIFY_PICKLE_TEST_LOC)