""" Data parsing. Mainly borrowed from previous project using the EC3 dataset. 
Source code at: https://github.com/dailykirt/ML_Enron_email_summary/blob/master/notebooks/Process_Emails.ipynb
"""
import sys
import re
import numpy as np
import xml.etree.ElementTree as ET # https://docs.python.org/3/library/xml.etree.elementtree.html
import pandas as pd
from talon.signature.bruteforce import extract_signature # https://pypi.org/project/talon/
import email
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

parsedEmailXML = ET.parse( "./data/bc3/corpus.xml" )
parsedAnnotationXML = ET.parse( "./data/bc3/annotation.xml" )
email_root = parsedEmailXML.getroot()
annotation_root = parsedAnnotationXML.getroot()

""" 
Extracts the dc3 email data from original XML format
"""
def parseDc3Emails(root):
    BC3_email_list = []
    #The emails are seperated by threads.
    for thread in root:
        email_num = 0
        #Iterate through the thread elements <name, listno, Doc>
        for thread_element in thread:
            #Getting the listno allows us to link the summaries to the correct emails
            if thread_element.tag == "listno":
                listno = thread_element.text
            #Each Doc element is a single email
            if thread_element.tag == "DOC":
                email_num += 1
                email_metadata = []
                for email_attribute in thread_element:
                    #If the email_attri is text, then each child contains a line from the body of the email
                    if email_attribute.tag == "Text":
                        email_body = ""
                        for sentence in email_attribute:
                            email_body += sentence.text
                    else:
                        #The attributes of the Email <Recieved, From, To, Subject, Text> appends in this order. 
                        email_metadata.append(email_attribute.text)
                        
                #Use same enron cleaning methods on the body of the email
                split_body = clean_body(email_body)
                    
                email_dict = {
                    "listno" : listno,
                    "date" : process_date(email_metadata[0]),
                    "from" : email_metadata[1],
                    "to" : email_metadata[2],
                    "subject" : email_metadata[3],
                    "body" : split_body['body'],
                    "email_num": email_num
                }
                
                BC3_email_list.append(email_dict)           
    return pd.DataFrame(BC3_email_list)

""" 
Extracts the dc3 annotations from original XML format
"""
def parseDc3Summary(root):
    BC3_summary_list = []
    for thread in root:
        #Iterate through the thread elements <listno, name, annotation>
        for thread_element in thread:
            if thread_element.tag == "listno":
                listno = thread_element.text
            #Each Doc element is a single email
            if thread_element.tag == "annotation":
                for annotation in thread_element:
                #If the email_attri is summary, then each child contains a summarization line
                    if annotation.tag == "summary":
                        summary_dict = {}
                        for summary in annotation:
                            #Generate the set of emails the summary sentence belongs to (often a single email)
                            email_nums = summary.attrib['link'].split(',')
                            s = set()
                            for num in email_nums:
                                s.add(num.split('.')[0].strip()) 
                            #Remove empty strings, since they summarize whole threads instead of emails. 
                            s = [x for x in set(s) if x]
                            for email_num in s:
                                if email_num in summary_dict:
                                    summary_dict[email_num] += ' ' + summary.text
                                else:
                                    summary_dict[email_num] = summary.text
                    #get annotator description
                    elif annotation.tag == "desc":
                        annotator = annotation.text
                #For each email summarizaiton create an entry
                for email_num, summary in summary_dict.items():
                    email_dict = {
                        "listno" : listno,
                        "annotator" : annotator,
                        "email_num" : email_num,
                        "summary" : summary
                    }      
                    BC3_summary_list.append(email_dict)
    return pd.DataFrame(BC3_summary_list)

"""
Converts the MIME date format to a more pandas friendly type. 
"""
def process_date(date_time):
    '''
    Converts the MIME date format to a more pandas friendly type. 
    '''
    try:
        date_time = email.utils.format_datetime(email.utils.parsedate_to_datetime(date_time))
    except:
        date_time = None
    return date_time

"""
This extracts both the email signature, and the forwarding email chain if it exists. 
"""
def clean_body(mail_body):
    delimiters = ["-----Original Message-----","To:","From"]
    
    #Trying to split string by biggest delimiter. 
    old_len = sys.maxsize
    
    for delimiter in delimiters:
        split_body = mail_body.split(delimiter,1)
        new_len = len(split_body[0])
        if new_len <= old_len:
            old_len = new_len
            final_split = split_body
            
    #Then pull chain message
    if (len(final_split) == 1):
        mail_chain = None
    else:
        mail_chain = final_split[1] 
    
    #The following uses Talon to try to get a clean body, and seperate out the rest of the email. 
    clean_body, sig = extract_signature(final_split[0])
    
    return {'body': clean_body, 'chain' : mail_chain, 'signature': sig}
"""
These remove symbols and character patterns that don't aid in producing a good summary. 
"""
def clean_email_df(df):
    #Removing strings related to attatchments and certain non numerical characters.
    patterns = ["\[IMAGE\]","-", "_", "\*", "+","\".\""]
    for pattern in patterns:
        df['body'] = pd.Series(df['body']).str.replace(pattern, "")
    
    #Remove multiple spaces. 
    df['body'] = df['body'].replace('\s+', ' ', regex=True)

    #Blanks are replaced with NaN in the whole dataframe. Then rows with a 'NaN' in the body will be dropped. 
    df = df.replace('',np.NaN)
    df = df.dropna(subset=['body'])

    #Remove all Duplicate emails 
    #df = df.drop_duplicates(subset='body')
    return df

"""
This function removes stopwords
"""
def remove_stopwords(sen):
    stop_words = stopwords.words('english')
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

"""
This function splits up the body into sentence tokens and removes stop words. 
"""
def tokenize_email(text):
    clean_sentences = sent_tokenize(text, language='english')
    #removing punctuation, numbers and special characters. Then lowercasing. 
    clean_sentences = [re.sub('[^a-zA-Z ]', '',s) for s in clean_sentences]
    clean_sentences = [s.lower() for s in clean_sentences]
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    return clean_sentences
"""
STEP 1: Collecting and structuring the bc3 data
"""

bc3_email_df = parseDc3Emails(email_root)
bc3_summary_df = parseDc3Summary(annotation_root)
bc3_summary_df['email_num'] = bc3_summary_df['email_num'].astype(int) # needed to be able to merge the pandas!

# bc3_email_df.info()
# bc3_summary_df.info()

# merge the dataframes together
bc3_df = pd.merge(bc3_email_df, 
                  bc3_summary_df[['annotator', 'email_num', 'listno', 'summary']],
                 on=['email_num', 'listno'])

# print out the dataframes info.
# bc3_df.info()
# print(bc3_df.head())

"""
STEP 2: Pre-processing of the input data
"""

# set to pandas datetime (for easier access and handling)
bc3_df['date'] = pd.to_datetime(bc3_df.date, utc=True)

# Example use of the date when in pandas format
""" start_date = str(bc3_df.date.min())
end_date =  str(bc3_df.date.max())
print("Start Date: " + start_date)
print("End Date: " + end_date) """

# Sentence cleaning
bc3_df = clean_email_df(bc3_df)

# Tokenizing
# TODO: check if we actually want to remove stop words! Seems like a good idea to get the vector sizes down though..
bc3_df['extractive_sentences'] = bc3_df['body'].apply(sent_tokenize)
bc3_df['tokenized_body'] = bc3_df['body'].apply(tokenize_email)
print(bc3_df.head())

""" 
STEP 3: Store data
"""
BC3_PICKLE_LOC  = "./data/dataframes/wrangled_BC3_df.pkl"
bc3_df.to_pickle(BC3_PICKLE_LOC)
