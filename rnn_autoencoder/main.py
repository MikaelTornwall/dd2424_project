import pandas as pd
from seq2seq import *
from language import *


def main():
    data = pd.read_pickle(r'../data/dataframes/wrangled_BC3_df.pkl')
    print(data.info())

    X = data['body']
    Y = data['summary']


    clean_body = clean_text(X)
    clean_summary = clean_text(Y)

    # print(data['tokenized_body'][0])
    input_language, output_language, pairs = prepare_data(clean_body, clean_summary)

    input_tensor, target_tensor = tensor_from_pair(input_language, output_language, pairs[0])
    print(input_tensor)
    print(target_tensor)


if __name__ == '__main__':
    main()