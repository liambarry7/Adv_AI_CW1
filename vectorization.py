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
    training_set.to_csv(f"{folder}//{target_csv}_train.csv", index=False)
    test_set.to_csv(f"{folder}//{target_csv}_test.csv", index=False)

    # Save the vectorizer using pickle
    with open(f'{folder}//{target_csv}_bow_v.pkl', 'wb') as file:
        pickle.dump(bow_v, file)

    # save the feature vectors
    # https://docs.scipy.org/doc/scipy/reference/sparse.html
    save_npz(f"{folder}//{target_csv}_train.npz", bow_tokens_train)
    save_npz(f"{folder}//{target_csv}_test.npz", bow_tokens_test)


def nlu_tfidf(v_type, target_csv, folder):
    # get csv
    df = pd.read_csv(f"{folder}//{target_csv}.csv")

    # check for any nans
    nan_rows = df[df.isna().any(axis=1)]
    print(f"nan rows: {nan_rows}")
    ready_df = df.dropna().reset_index(drop=True)

    nan_rows2 = ready_df[ready_df.isna().any(axis=1)]
    print(f"nan rows: {nan_rows2}")

    # create one long vector of headlines and tokens
    ready_df['full_tokens'] = ready_df["headline_tokens"] + " " + ready_df["post_tokens"]


    # create TFIDF vectorizer
    tfidf_v = TfidfVectorizer(min_df=10)

    # train vector on all tokens
    tfidf_v.fit_transform(ready_df['full_tokens']) # train on full data

    # create vectors for headlines and posts
    headline_tfidf = tfidf_v.transform(ready_df['headline_tokens'])
    post_tfidf = tfidf_v.transform(ready_df['post_tokens'])

    # save vectorizer, and vectors
    with open(f'{folder}//tfidf_v.pkl', 'wb') as file:
        pickle.dump(tfidf_v, file)

    # save_npz(f"{folder}//{target_csv}_full.npz", )
    save_npz(f"{folder}//{target_csv}_tfidf_headline.npz", headline_tfidf)
    save_npz(f"{folder}//{target_csv}_tfidf_post.npz", post_tfidf)

def nlu_bow(v_type, target_csv, folder):
    # get csv
    df = pd.read_csv(f"{folder}//{target_csv}.csv")

    # check for any nans
    nan_rows = df[df.isna().any(axis=1)]
    print(f"nan rows: {nan_rows}")
    ready_df = df.dropna().reset_index(drop=True)

    nan_rows2 = ready_df[ready_df.isna().any(axis=1)]
    print(f"nan rows: {nan_rows2}")

    # create one long vector of headlines and tokens
    ready_df['full_tokens'] = ready_df["headline_tokens"] + " " + ready_df["post_tokens"]


    # create TFIDF vectorizer
    bow_v = CountVectorizer(min_df=10)

    # train vector on all tokens
    bow_v.fit_transform(ready_df['full_tokens']) # train on full data

    # create vectors for headlines and posts
    headline_bow = bow_v.transform(ready_df['headline_tokens'])
    post_bow = bow_v.transform(ready_df['post_tokens'])

    # save vectorizer, and vectors
    with open(f'{folder}//bow_v.pkl', 'wb') as file:
        pickle.dump(bow_v, file)

    # save_npz(f"{folder}//{target_csv}_full.npz", )
    save_npz(f"{folder}//{target_csv}_bow_headline.npz", headline_bow)
    save_npz(f"{folder}//{target_csv}_bow_post.npz", post_bow)

def tfidf(target_csv, column, folder, raw=0):
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

    # create TFIDF vectorizer
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
    training_set.to_csv(f"{folder}//{target_csv}_train.csv", index=False)
    test_set.to_csv(f"{folder}//{target_csv}_test.csv", index=False)

    # Save the vectorizer using pickle
    with open(f'{folder}//{target_csv}_tfidf_v.pkl', 'wb') as file:
        pickle.dump(tfidf_v, file)

    # save the feature vectors
    # https://docs.scipy.org/doc/scipy/reference/sparse.html
    save_npz(f"{folder}//{target_csv}_train.npz", tfidf_train)
    save_npz(f"{folder}//{target_csv}_test.npz", tfidf_test)


def combine_datasets(v_type, file1, file2, columns, name):
    #   AMEND TO FIX FOR RAW DATASETS RATHER THAN ALRADY VECTORISED ONES

    # for i in ["train", "test"]:
    #     # combine two datasets together where doc ID match
    df1 = pd.read_csv(f"{v_type}//{file1}.csv")
    df2 = pd.read_csv(f"{v_type}//{file2}.csv")

    combined = pd.merge(df1, df2, on=["id", "class_label"], how="inner")
    print(combined.columns)
    combined['tokens'] = combined[columns[0]] + " " + combined[columns[1]]
    combined.drop(columns=columns, inplace=True)
    combined.to_csv(f"{v_type}//{name}.csv", index=False)




def test():
    # https://www.geeksforgeeks.org/nlp/vectorization-techniques-in-nlp/
    # https://www.geeksforgeeks.org/nlp/how-to-store-a-tfidfvectorizer-for-future-use-in-scikit-learn/
    # https://docs.scipy.org/doc/scipy/reference/sparse.html



    pass

def create_tfidf():
    # create lem and stem tokens
    tfidf("post-tokens_lem", "post_tokens", "tfidf")
    tfidf("post-tokens_stem", "post_tokens", "tfidf")

    # # create ner and pos vectors
    # tfidf("post-entities", "entities", "tfidf")
    tfidf("post-pos", "pos", "tfidf")

    # stem == better dataset (check lem vs stem csv), quicker and same accuracy
    # combine datasets -> add csv files together, then vectorize
    combine_datasets("raw_datasets", "post-tokens_stem", "post-pos", ["post_tokens", "pos"], "tokens_pos")

    # create token-pos vector
    tfidf("tokens_pos", "tokens", "tfidf")

def create_bow():
    # create lem and stem tokens
    bow("post-tokens_lem", "post_tokens", "bow")
    bow("post-tokens_stem", "post_tokens", "bow")

    # # create ner and pos vectors
    # tfidf("post-entities", "entities", "tfidf")
    bow("post-pos", "pos", "bow")

    # stem == better dataset (check lem vs stem csv), quicker and same accuracy
    # combine datasets -> add csv files together, then vectorize
    combine_datasets("raw_datasets", "post-tokens_stem", "post-pos", ["post_tokens", "pos"], "tokens_pos")

    # create token-pos vector
    bow("tokens_pos", "tokens", "bow")


if __name__ == "__main__":
    # test()

    # task one
    # create_bow()
    # create_tfidf()

    # task two
    # nlu_tfidf("tfidf", "tokens_lem", "nlu_data")
    nlu_bow("bow", "tokens_lem", "nlu_data")

