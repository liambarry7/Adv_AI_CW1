from nlp import text_preprocess, get_pos
from vectorization import bow, tfidf, nlu_bow, combine_datasets
import pickle

def run():
    raw_file_name = input("Please enter the raw dataset file name: ")
    tokenise = input("Tokenize text? - y/n")
    if tokenise == "y":
        # tokenise the input
        text_preprocess(raw_file_name, "stem")

    vect = input("Vectorizing text? - y/n")
    if vect == "y":
        v = input("BoW or TFIDF? - b/t")
        if v == "b":
            bow("post-tokens_lem", "post_tokens", "bow")

            mlp = input("MLP model? - y/n")
            if mlp == "y":
                # load mlp model
                with open('best_bow_mlp.pkl', 'r') as file:
                    mlp = pickle.load(file)

                    # y_predictions = mlp.predict(test_x)
                    #
                    # print(y_predictions)
                    # print(f"Accuracy: {accuracy_score(test_y, y_predictions)}")
        else:
            tfidf("post-tokens_lem", "post_tokens", "tfidf")




        # predict data

if __name__ == "__main__":
    run()