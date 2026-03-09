from sklearn.neural_network import MLPClassifier
import pandas as pd
from scipy.sparse import load_npz


def mlp():
    mlp = MLPClassifier()

    training_set = pd.read_csv("bow//post-tokens_lem_training.csv")
    test_set = pd.read_csv("bow//post-tokens_lem_test.csv")

    train_y = training_set['class_label']
    test_y = test_set['class_label']

    train_x = load_npz("bow//post-tokens_lem_train.npz")
    test_x = load_npz("bow//post-tokens_lem_test.npz")

    mlp.fit(train_x.toarray(), train_y)
    y_predictions = mlp.predict(test_x.toarray())

    print(y_predictions)

def test():

    pass


if __name__ == "__main__":
    # test()
    mlp()
