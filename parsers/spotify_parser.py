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

# Read in the metadata. Separate out each episode into a dict
def read_spotify_metadata():
    print('reading metadata..')
    episode_dict = {}
    with open('../data/spotify/outmeta.json', 'r') as j:
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
    return episode_dict

def read_spotify_transcript_data(episode_dict):
    """
    Function that:
        Read in the transcript files. 
        mergeing all the transcripts to one text block (body)
        Connecting each text block to the description in the metadata
        Creating pd dataframe of columns 'body' 'description' 'episode_name' 'podcast_name'...
    Param: episode_dict mapping episode id to the episode metadata
    Return: dataframe containing episode objects
    """
    print('reading files..')
    training_data = []
    test_data = []
    dir_name = '../data/spotify/transformed/'   
    files = [os.path.join(dir_name, fn) for fn in os.listdir(dir_name)]
    num_files = 0
    num_files_tot = 0
    for file in files:
        if num_files >= 500: # only read 500 files for now..
            break
        num_files_tot += 1
        with open(file, 'r') as j:
            json_data = json.load(j)
            id = json_data['id']
            if id in episode_dict:
                ep_metadata = episode_dict[id]
                body_arr = []
                for item in json_data['data']:
                    if 'transcript' in item:
                        body_arr.append(item['transcript'])
                body = ' '.join(body_arr)
                episode_len = len(body)
                desc_len = len(ep_metadata['ep_desc'])
                # only include shorter episodes with shorter descriptions!
                if episode_len < 10000 and episode_len > 0 and desc_len < 100 and desc_len > 0:
                    num_files += 1
                    print('adding file wit ep len / sum len', episode_len, '/', desc_len)
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

    print('episodes checked: ', num_files_tot)
    df_spotify_train = pd.DataFrame(training_data)
    df_spotify_test = pd.DataFrame(test_data)
    print(df_spotify_train)
    print(df_spotify_test)
    return df_spotify_train, df_spotify_test

def clean_data(df_spotify_train, df_spotify_test):
    """
    Cleaning and tokenizing the data
    Parameters: train and test dataframes
    Returns: train and test dataframes with additional cleaned columns
    """
    print('cleaning data..')
    df_spotify_train['extractive_sentences'] = df_spotify_train['body'].apply(sent_tokenize)
    df_spotify_train['tokenized_body'] = df_spotify_train['body'].apply(tokenize_email)
    df_spotify_test['extractive_sentences'] = df_spotify_test['body'].apply(sent_tokenize)
    df_spotify_test['tokenized_body'] = df_spotify_test['body'].apply(tokenize_email)
    return df_spotify_train, df_spotify_test

def write_spotify_data(df_spotify_train, df_spotify_test):
    """ 
    Store data
    """
    SPOTIFY_PICKLE_TRAIN_LOC  = "../data/dataframes/spotify_df_train.pkl"
    SPOTIFY_PICKLE_TEST_LOC  = "../data/dataframes/spotify_df_test.pkl"
    df_spotify_train.to_pickle(SPOTIFY_PICKLE_TRAIN_LOC)
    df_spotify_test.to_pickle(SPOTIFY_PICKLE_TEST_LOC)

"""
Main function that reads, parses and writes the spotify data.
"""
def parse_spotify_data():
    episode_dict = read_spotify_metadata()
    df_spotify_train, df_spotify_test = read_spotify_transcript_data(episode_dict)
    df_spotify_train, df_spotify_test = clean_data(df_spotify_train, df_spotify_test)
    write_spotify_data(df_spotify_train, df_spotify_test)

parse_spotify_data()