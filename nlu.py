import pandas as pd
from vectorization import combine_datasets
"""
    - use NLU & NLP to explore and analyse content in docuemtns and their linked news headlines
    to automatically discover topics across text data
    - implement standard syntactic analysis, including parsing/tokenisation, stop word removal,
    lemmatization and stemming.
    - experiment with at least two text representation strategies (e.g. LDA, BoW, TFIDF,
    word vector/embeddings
    - evaluate results from models, discuss discovered topics and their link to document content,
    news headlines and truth labels
    - save the best model


    - tokenise as per nlp
    - then do topic extraction from tokens, create a document-topic matrix
    - look at week 4 lab
    -> vectorise the tokens, then put into LDA for topic extraction
    -> compare the effectiveness of different vectorisation techniques

    - get intent from sentences
    - use synonym lists


"""

def tokenise(mode="lem"):
    df = pd.read_csv("raw_datasets/social-media-release.csv")

    # print any rows with nan values
    nan_rows = df[df.isna().any(axis=1)]
    print(f"nan rows: {nan_rows}")

    # if row contains null value, remove
    df = df.dropna().reset_index(drop=True)

    # remove any duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    


def syntactic_analysis():
    # get tokens (already parsed, stop word removed etc)
    lem_t = pd.read_csv("raw_datasets//post-tokens_lem.csv")
    stem_t = pd.read_csv("raw_datasets//post-tokens_stem.csv")

    # tokenise headlines

    # turn tokens into word vectors

    # pass word vectors into lda to discover latent topics

    pass


def test():

    pass


if __name__ == "__main__":
    test()


