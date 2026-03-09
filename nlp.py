import spacy
import spacy.cli
from nltk.stem import PorterStemmer
import pandas as pd


def text_tokenizing_lem(text):
    tokens = []

    ps = PorterStemmer()
    doc = nlp(text.lower())  # lowercase string
    for token in doc:
        # ignore whitespace, punc, stop words and only accept words (e.g. not numbers, symbols)
        if not token.is_space and not token.is_stop and not token.is_punct and token.is_alpha:

            # lemmatization
            # print(f"{token.text} : {token.lemma_}")
            tokens.append(token.lemma_)

    print(f"Tokenisation complete: {tokens}")

    if len(tokens) == 0:
        return '0'
    else:
        return ' '.join(tokens)


def text_tokenizing_stem(text):
    tokens = []

    ps = PorterStemmer()
    doc = nlp(text.lower())  # lowercase string
    for token in doc:
        # ignore whitespace, punc, stop words and only accept words (e.g. not numbers, symbols)
        if not token.is_space and not token.is_stop and not token.is_punct and token.is_alpha:

            # stemming
            # print(f"{token.text} : {ps.stem(token.text)}")
            tokens.append(ps.stem(token.text))

    print(f"Tokenisation complete: {tokens}")
    # return ' '.join(tokens)
    if len(tokens) == 0:
        return '0'
    else:
        return ' '.join(tokens)

def text_preprocess(mode="lem"):
    # tokenise text and save to csv file

    df = pd.read_csv("raw_datasets/social-media-release.csv")

    # print any rows with nan values
    nan_rows = df[df.isna().any(axis=1)]
    print(f"nan rows: {nan_rows}")

    # if row contains null value, remove
    df = df.dropna().reset_index(drop=True)

    # remove any duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    nan_rows = df[df.isna().any(axis=1)]
    print(f"nan rows: {nan_rows}")

    if mode == "stem":
        df['post_tokens'] = df['post'].apply(text_tokenizing_stem)

    else:
        df['post_tokens'] = df['post'].apply(text_tokenizing_lem)

    # encode class label
    df['class_label'] = df['class_label'].astype('category').cat.codes

    nan_rows = df[df.isna().any(axis=1)]
    print(f"nan rows: {nan_rows}")

    # if row has no tokens, remove row
    df_cleaned = df.dropna()

    df_final = df_cleaned[['id', 'class_label', 'post_tokens']]

    df_final.to_csv(f"raw_datasets//post-tokens_{mode}.csv", index=False)

def ner(text):
    # returns all the different types of entities (e.g. "DATE", "ORG" etc)
    tokens = []
    doc = nlp(text)

    for token in doc:
        if token.ent_type_ != "":  # token belongs to an entity
            # print(
            #     f"{str(token.text):20}",
            #     f"{str(token.ent_type_):8}",
            #     f"{str(spacy.explain(token.ent_type_))}"
            # )
            tokens.append(token.ent_type_)

    print(f"Entities: {tokens}")
    # return ' '.join(tokens)

    if len(tokens) == 0:
        return '0'
    else:
        return ' '.join(tokens)

def get_entities():
    # get text entities and save to csv file

    df = pd.read_csv("raw_datasets/social-media-release.csv")

    # print any rows with nan values
    nan_rows = df[df.isna().any(axis=1)]
    print(f"nan rows: {nan_rows}")

    # if row contains null value, remove
    df = df.dropna().reset_index(drop=True)

    # remove any duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    nan_rows = df[df.isna().any(axis=1)]
    print(f"nan rows: {nan_rows}")

    df['entities'] = df['post'].apply(ner)

    # encode class label
    df['class_label'] = df['class_label'].astype('category').cat.codes

    nan_rows = df[df.isna().any(axis=1)]
    print(f"nan rows: {nan_rows}")

    # if row has no tokens, remove row
    df_cleaned = df.dropna()

    df_final = df_cleaned[['id', 'class_label', 'entities']]

    df_final.to_csv(f"raw_datasets//post-entities.csv", index=False)



def test():

    print("Test harness")

    df = pd.read_csv("raw_datasets//social-media-release.csv")
    print(df.head())

    # get first 3 rows
    test_df = df.head(3)

    for post in test_df['post']:
        text_tokens_lem = text_tokenizing_lem(post)
        text_tokens_stem = text_tokenizing_stem(post)

        ner_entity_types = ner(post)

        # pos_tags = part_of_speech(post)

        print(f"\nPost : {post}")
        print(f"Tokens lem : {text_tokens_lem}")
        print(f"Tokens stem : {text_tokens_stem}")
        print(f"Entity types : {ner_entity_types}")
        # print(f"POS tags {pos_tags}")
        print("\n")

if __name__ == "__main__":
    # spacy.cli.download("en_core_web_sm")
    nlp = spacy.load('en_core_web_sm')
    # test()

    # text_preprocess("lem")
    text_preprocess("stem")
    # get_entities()