from sklearn.metrics import accuracy_score

from nlp import text_preprocess, tokens_for_topics
from vectorization import bow, tfidf, nlu_bow, nlu_tfidf
import pickle
import pandas as pd
from scipy.sparse import load_npz
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def run(raw_file_name):
    tokenise = input("Tokenize text? - y/n")
    if tokenise == "y":
        # tokenise the input
        text_preprocess(raw_file_name, "stem")

    vect = input("Vectorizing text? - y/n")
    if vect == "y":
        v = input("BoW or TFIDF? - b/t")
        if v == "b":
            bow("post-tokens_lem", "post_tokens", "bow")
        else:
            tfidf("post-tokens_lem", "post_tokens", "tfidf")




def run_mlp(v_type, target_csv):
    with open(f'best_{v_type}_mlp.pkl', 'r') as file:
        mlp_model = pickle.load(file)

    test_set = pd.read_csv(f"{v_type}//{target_csv}_test.csv")
    test_y = test_set["class_label"]

    test_x = load_npz(f"{v_type}//{target_csv}_test.npz")

    y_predictions = mlp_model.predict(test_x)

    print(y_predictions)
    print(f"Accuracy: {accuracy_score(test_y, y_predictions)}")

def run_cnn(v_type, target_csv):
    with open('best_bow_cnn.pkl', 'r') as file:
        cnn_model = pickle.load(file)

    training_set = pd.read_csv(f"{v_type}/{target_csv}_train.csv")
    test_set = pd.read_csv(f"{v_type}/{target_csv}_test.csv")

    train_y = training_set["class_label"]
    test_y = test_set["class_label"]

    train_x = load_npz(f"{v_type}/{target_csv}_train.npz").toarray()
    test_x = load_npz(f"{v_type}/{target_csv}_test.npz").toarray()

    print(test_x.shape)

    max_len = train_x.shape[1]

    # expects 3D tensor = (Batch size, sequence length, embedding dimensions)
    train_x = train_x.reshape(train_x.shape[0], max_len, 1)
    test_x = test_x.reshape(test_x.shape[0], max_len, 1)

    batch_size = 128  # set the batch size for updating weights
    num_epochs = 5  # train the model for 5 epochs
    # train the model, set "verbose=1" to show the training process
    model_log = cnn_model.fit(train_x, train_y, batch_size=batch_size,
                              epochs=num_epochs, validation_data=(test_x, test_y), verbose=1)

    score = cnn_model.evaluate(test_x, test_y, verbose=1)
    print(f"Test loss: {score[0]}")
    print(f"Test accuracy: {score[1]}")

def run_topics(v_type, target_csv):
    # tokenise for topics
    tokens_for_topics(target_csv, "lem")
    # create vectors
    if v_type == "bow":
        nlu_bow("bow", "tokens_lem", "nlu_data")
    elif v_type == "tfidf":
        nlu_tfidf("tfidf", "tokens_lem", "nlu_data")

    headlines = load_npz(f"nlu_data/tokens_lem_{v_type}_headline.npz")
    posts = load_npz(f"nlu_data/tokens_lem_{v_type}_post.npz")

    # get model
    with open('nlu_data//lda_bow_2_50.pkl', 'r') as file:
        lda_model = pickle.load(file)

    # get topics from headlines and posts
    print("Transforming headlines and posts...")
    headlines_topics = lda_model.transform(headlines)
    posts_topics = lda_model.transform(posts)


    # print topic-word distribution
    print(lda_model.components_)
    print(lda_model.components_.shape)  # (n_topics, vocabulary size)

    # print_top_words(lda_model, "x", 0, 10)

    print("Calculating headline-post topic similarity...")
    # compare using cosine similarity
    similarities = np.array(
        [cosine_similarity(headlines_topics[i].reshape(1, -1), posts_topics[i].reshape(1, -1))[0][0] for i in
         range(headlines_topics.shape[0])])
    print(similarities.shape)
    print(similarities[:500])

    avg_similarity = np.mean(similarities)
    print(f"average similarity: {avg_similarity}")
    print(f"min similarity: {np.min(similarities)}")
    print(f"max similarity: {np.max(similarities)}")
    print(f"median similarity: {np.median(similarities)}")


if __name__ == "__main__":
    # run -> give raw file name, will tokenise and vectorise
    # run models -> give v_type and target file name, returns accuracy from pre-trained model
    run()
    run_mlp()
    run_cnn()
    run_topics("bow", "nlu_data")