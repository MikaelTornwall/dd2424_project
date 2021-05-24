# DD2424 text summarization project

Text summarization using deep neural networks.

Authors:
Isak Pettersson
Filippa Kärrfelt
Mikael Törnwall

## Installation

Install the project dependencies, run

```bash
pip install -r requirements.txt
```

Or other ways to install the necessary dependencies.

For the nltk modules to work data need to be downloaded and stored. Run
```bash
nltk.download('punkt')
nltk.download('stopwords')
```
Further instructions and information will be provided in the terminal when running the bc3_parser file.

## Usage

### Parsing and cleaning the data

create 'data' folder and the subfolders 'bc3' and 'dataframes'. The bc3 XML data should be placed in the bc3 folder.

```bash
python bc3_parser.py
```

The pandas dataframe will be written to the file 'wrangled_BC3_df.pkl'.

## Creating the word vectors
create the subfolder'wordvectors' in the data folder. The glove word vectors (with 50 dimensions) should be placed in the wordvectors folder. glove word vectors can be downloaded from: http://nlp.stanford.edu/data/glove.6B.zip

To compute the sentence vectors, both based on term frequency and glove word vectors, run:

```bash
python sentence_vectors.py
```

The pandas dataframe, including the vectors will be written to the file 'BC3_df_with_sentence_vectors.pkl'.

## Autoencoder
The functionality for the autoencoder is in the autoencoder module. The autoencoder is training on all sentences in the corpus, and can then output sentence vectors based on the grove sentence vectors for a document.

## Sequence2Sequence network

## Summarization
The project includes functionality to summarize documents, based on textrank scores of the sentences in a document. Summarization functionality is located in the file summarize.py

## Evaluation
The project includes functionality to evaluate summaries created based on textrank. The summaries are scored using ROUGE. Scores are computed both on the summaries created using textrank on the original glove sentence vectors and on the glove sentence vectors returned by the autoencoder. To run the evaluation: 

```bash
python evaluate.py
```
