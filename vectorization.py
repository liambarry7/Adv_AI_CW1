import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import save_npz, load_npz, hstack

def dataset_split(df):
    # split dataset into training and test sets
    print("Split dataset to test, training")
    # df = pd.read_csv(df)
    training_set, test_set = train_test_split(df, random_state=42, test_size=0.2)
    print(training_set.shape, test_set.shape)

    return training_set, test_set

def bow(target_csv, column, folder):
    # Bag-of-Words

    # target_csv = file being read (e.g. post-tokens-lem)
    # column = column being vectorized
    # folder = target folder for files to be saved to

    df = pd.read_csv(f"raw_datasets//{target_csv}.csv")
    print(df.head())

    # check for any nans
    nan_rows = df[df.isna().any(axis=1)]
    print(f"nan rows: {nan_rows}")
    ready_df = df.dropna().reset_index(drop=True)

    nan_rows2 = ready_df[ready_df.isna().any(axis=1)]
    print(f"nan rows: {nan_rows2}")

    # split the dataset before vectorization
    training_set, test_set = dataset_split(ready_df)

    # create BoW vectorizer
    bow_v = CountVectorizer(min_df=10) # only include words that appear in 10 > documents

    # fit_transform the training data
    bow_tokens_train = bow_v.fit_transform(training_set[column])

    # only transform the test data as dont want vectorizor to learn the test data too
    bow_tokens_test = bow_v.transform(test_set[column])

    print(len(bow_v.get_feature_names_out()))
    print(type(bow_tokens_train))

    # print(f"training bow: {bow_tokens_train}")
    # print(f"test bow: {bow_tokens_test}")

    # save the datasets, feature vectors and vectorizer
    # save the datasets
    training_set.to_csv(f"{folder}//{target_csv}_training.csv", index=False)
    test_set.to_csv(f"{folder}//{target_csv}_test.csv", index=False)

    # Save the vectorizer using pickle
    with open(f'{folder}//{target_csv}_bow_v.pkl', 'wb') as file:
        pickle.dump(bow_v, file)

    # save the feature vectors
    # https://docs.scipy.org/doc/scipy/reference/sparse.html
    save_npz(f"{folder}//{target_csv}_train.npz", bow_tokens_train)
    save_npz(f"{folder}//{target_csv}_test.npz", bow_tokens_test)

def tfidf(target_csv, column, folder):
    # target_csv = file being read (e.g. post-tokens-lem)
    # column = column being vectorized
    # folder = target folder for files to be saved to

    df = pd.read_csv(f"raw_datasets//{target_csv}.csv")
    print(df.head())

    # check for any nans
    nan_rows = df[df.isna().any(axis=1)]
    print(f"nan rows: {nan_rows}")
    ready_df = df.dropna().reset_index(drop=True)

    nan_rows2 = ready_df[ready_df.isna().any(axis=1)]
    print(f"nan rows: {nan_rows2}")

    # split the dataset before vectorization
    training_set, test_set = dataset_split(ready_df)

    # create BoW vectorizer
    tfidf_v = TfidfVectorizer(min_df=10) # only include words that appear in 10 > documents

    # fit_transform the training data - context week 4 lab part c
    tfidf_train = tfidf_v.fit_transform(training_set[column])

    # only transform the test data as dont want vectorizor to learn the test data too
    tfidf_test = tfidf_v.transform(test_set[column])

    print(type(tfidf_train))
    print(len(tfidf_v.get_feature_names_out()))

    # print(f"training bow: {tfidf_train}")
    # print(f"test bow: {tfidf_test}")

    # save the datasets, feature vectors and vectorizer
    # save the datasets
    training_set.to_csv(f"{folder}//{target_csv}_training.csv", index=False)
    test_set.to_csv(f"{folder}//{target_csv}_test.csv", index=False)

    # Save the vectorizer using pickle
    with open(f'{folder}//{target_csv}_tfidf_v.pkl', 'wb') as file:
        pickle.dump(tfidf_v, file)

    # save the feature vectors
    # https://docs.scipy.org/doc/scipy/reference/sparse.html
    save_npz(f"{folder}//{target_csv}_train.npz", tfidf_train)
    save_npz(f"{folder}//{target_csv}_test.npz", tfidf_test)

def combine_datasets(v_type):
    # combine datasets from a list
    print(f"Combine {v_type} datasets...")
    # combine tokens + ner for example using hstack?
    # create a long vector with more data
    set = ["train", "test"]

    # get training data
    for t in set:

        tokens_lem = load_npz(f"{v_type}//post-tokens_lem_{t}.npz")
        tokens_stem = load_npz(f"{v_type}//post-tokens_lem_{t}.npz")
        entities = load_npz(f"{v_type}//post-entities_{t}.npz")
        pos = load_npz(f"{v_type}//post-pos_{t}.npz")

        lem_ner = hstack([tokens_lem, entities])
        stem_ner = hstack([tokens_stem, entities])
        lem_pos = hstack([tokens_lem, pos])
        stem_pos = hstack([tokens_stem, pos])
        lem_ner_pos = hstack([tokens_lem, entities, pos])
        stem_ner_pos = hstack([tokens_stem, entities, pos])

        save_npz(f"{v_type}//lem_ner_{t}.npz", lem_ner)
        save_npz(f"{v_type}//stem_ner_{t}.npz", stem_ner)
        save_npz(f"{v_type}//lem_pos_{t}.npz", lem_pos)
        save_npz(f"{v_type}//stem_pos_{t}.npz", stem_pos)
        save_npz(f"{v_type}//lem_ner_pos_{t}.npz", lem_ner_pos)
        save_npz(f"{v_type}//stem_ner_pos_{t}.npz", stem_ner_pos)



def test():
    # https://www.geeksforgeeks.org/nlp/vectorization-techniques-in-nlp/
    # https://www.geeksforgeeks.org/nlp/how-to-store-a-tfidfvectorizer-for-future-use-in-scikit-learn/
    # https://docs.scipy.org/doc/scipy/reference/sparse.html



    pass

if __name__ == "__main__":
    # test()
    # bow("post-tokens_lem", "post_tokens", "bow")
    # bow("post-tokens_stem", "post_tokens", "bow")
    # bow("post-entities", "entities", "bow")
    # bow("post-pos", "pos", "bow")
    #
    # tfidf("post-tokens_lem", "post_tokens", "tfidf")
    # tfidf("post-tokens_stem", "post_tokens", "tfidf")
    # tfidf("post-entities", "entities", "tfidf")
    # tfidf("post-pos", "pos", "tfidf")

    combine_datasets("bow")
    combine_datasets("tfidf")