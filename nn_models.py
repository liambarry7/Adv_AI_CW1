from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
import pandas as pd
from scipy.sparse import load_npz

import tensorflow as tf
print(f"tensorflow:{tf.__version__}")

from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Model, Sequential

import time

def mlp(target_csv, column):

    start = time.time()

    # mlp = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=1000, early_stopping=True)
    mlp = MLPClassifier(max_iter=1000, early_stopping=True)

    training_set = pd.read_csv(f"{target_csv}_training.csv")
    test_set = pd.read_csv(f"{target_csv}_test.csv")

    train_y = training_set[column]
    test_y = test_set[column]

    train_x = load_npz(f"{target_csv}_train.npz")
    test_x = load_npz(f"{target_csv}_test.npz")

    mlp.fit(train_x, train_y)

    end = time.time()
    print(f"Fit data time taken: {end - start}")

    start = time.time()

    y_predictions = mlp.predict(test_x)

    print(y_predictions)
    print(f"Accuracy: {accuracy_score(test_y, y_predictions)}")

    end = time.time()
    print(f"Predictions time taken: {end - start}")

"""
# dictionary of hyperparams to test
    mlp_hyperparams = [{
        'hidden_layer_sizes': (100, 100),
        'activation': ('relu', 'logistic'),
        'learning_rate': ('constant', 'adaptive')
    }]
"""


def mlp_finetune(target_csv, column, vectorizerType):

    start = time.time()


    # get datasets
    training_set = pd.read_csv(f"{vectorizerType}//{target_csv}_train.csv")
    test_set = pd.read_csv(f"{vectorizerType}//{target_csv}_test.csv")

    train_y = training_set[column]
    test_y = test_set[column]

    train_x = load_npz(f"{vectorizerType}//{target_csv}_train.npz")
    test_x = load_npz(f"{vectorizerType}//{target_csv}_test.npz")

    # create a new mlp()
    mlp = MLPClassifier(max_iter=1000, early_stopping=True)

    # define 3 hyperparams to fine-tune
    mlp_hyperparams = [{
        'hidden_layer_sizes': [(100, 50), (64,32), (100,)],
        'activation': ['relu', 'tanh'],
        'learning_rate': ['constant', 'adaptive']
    }]

    # create 5-fold for StratifiedKFold to avoid imbalanced class distribution
    ff = StratifiedKFold(n_splits=5, shuffle=True, random_state=41)

    mlp_gridsearch = GridSearchCV(mlp, mlp_hyperparams, cv=ff, scoring='accuracy', refit=True, n_jobs=-1, verbose=3)
    # n_jobs = -1 : run on all available cores
    # verbose = 2 : gives update each time fold finishes

    # fit data to model
    mlp_gridsearch.fit(train_x, train_y)

    print(f"Best MLP params: {mlp_gridsearch.best_params_}")
    # print(mlp_gridsearch.scoring)
    print(f"Best MLP Object: {mlp_gridsearch.best_estimator_}")
    print(f"Best MLP Accuracy Score: {mlp_gridsearch.best_score_}")
    # print(mlp_gridsearch.cv_results_)

    cv_res = mlp_gridsearch.cv_results_
    print(cv_res.keys())  # dict_keys(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time', 'param_activation', 'param_hidden_layer_sizes', 'param_learning_rate', 'param_solver', 'params', 'split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score', 'mean_test_score', 'std_test_score', 'rank_test_score'])

    end = time.time()
    print(f"Time taken: {end - start}")


    results_df = pd.DataFrame(mlp_gridsearch.cv_results_)
    # # params = params used, mean_test_score = avg score over 5 folds, std_test_score =
    # results_df = results_df[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']].sort_values(by='rank_test_score')
    results_df = results_df[
        ['param_activation', 'param_hidden_layer_sizes', 'param_learning_rate', 'mean_fit_time', 'mean_test_score',
         'std_test_score', 'rank_test_score']].sort_values(by='rank_test_score')
    print(results_df.head())

    results_df.to_csv(f'ft_results//mlp_ft_{vectorizerType}_{target_csv}.csv', index=False)

def lem_vs_stem(v_type):
    # use cross validation to select lem or stem
    datasets = ["lem", "stem"]
    results = []

    for d in datasets:
        print(f"\nTesting {d}...")
        training_set = pd.read_csv(f"{v_type}//post-tokens_{d}_train.csv")

        train_y = training_set["class_label"]

        train_x = load_npz(f"{v_type}//post-tokens_{d}_train.npz")

        mlp = MLPClassifier(max_iter=1000, early_stopping=True)

        ff = StratifiedKFold(n_splits=5, shuffle=True, random_state=41)  # use StratifiedKFold to avoid imbalanced class distribution

        start = time.time()

        scores = cross_val_score(mlp, train_x, train_y, cv=ff, n_jobs=-1)

        end = time.time()
        total_time = end - start

        mean_cv = scores.mean()
        std_cv = scores.std()
        print(f"CV accuracy scores: {scores}")
        print(f"Mean CV accuracy: {mean_cv}")
        print(f"Standard deviation: {std_cv}")

        rs = {"v_type" : v_type,
               "name" : d,
               "mean_cv_accuracy" : mean_cv,
               "std" : std_cv,
              "total_time": total_time}

        results.append(rs)


    df = pd.DataFrame(results)
    df.to_csv(f"{v_type}_lem_vs_stem.csv")

def best_dataset(v_type):
    # use cross validation to select the best dataset
    datasets = ["post-tokens_lem", "post-tokens_stem", "post-pos", "tokens_pos"]
    results = []

    for d in datasets:
        print(f"\nTesting {d}...")
        training_set = pd.read_csv(f"{v_type}//{d}_train.csv")
        # test_set = pd.read_csv(f"{v_type}//{d}_test.csv")

        train_y = training_set["class_label"]
        # test_y = test_set["class_label"]

        train_x = load_npz(f"{v_type}//{d}_train.npz")
        # test_x = load_npz(f"{v_type}//{d}_test.npz")

        mlp = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=1000, early_stopping=True)

        ff = StratifiedKFold(n_splits=5, shuffle=True, random_state=41)  # use StratifiedKFold to avoid imbalanced class distribution

        start = time.time()

        scores = cross_val_score(mlp, train_x, train_y, cv=ff, n_jobs=-1)

        end = time.time()
        total_time = end - start

        mean_cv = scores.mean()
        std_cv = scores.std()
        print(f"CV accuracy scores: {scores}")
        print(f"Mean CV accuracy: {mean_cv}")
        print(f"Standard deviation: {std_cv}")
        print(f"Total time: {total_time}")

        rs = {"v_type" : v_type,
               "name" : d,
               "mean_cv_accuracy" : mean_cv,
               "std" : std_cv,
              "total_time": total_time}

        results.append(rs)


    df = pd.DataFrame(results)
    df.to_csv(f"{v_type}_dataset_comparison_mlp.csv")

def cnn(vectorizerType, target_csv, column):
    training_set = pd.read_csv(f"{vectorizerType}/{target_csv}_train.csv")
    test_set = pd.read_csv(f"{vectorizerType}/{target_csv}_test.csv")

    train_y = training_set[column]
    test_y = test_set[column]

    train_x = load_npz(f"{vectorizerType}/{target_csv}_train.npz")
    test_x = load_npz(f"{vectorizerType}/{target_csv}_test.npz")

    print(test_x.shape)

    # instantiate a CNN model, Sequential type
    # cnn_model = Sequential()
    # # convolutional layer with 32 3x3-filters, and ReLU activation function
    # cnn_model.add(Conv2D(32, kernel_size=(3, 3), activation="relu",
    #                      input_shape=(28, 28, 1)))
    # # stack a MaxPooling layer with 2x2 pool size following the first conv layer
    # cnn_model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    # # stack a Conv2D layer, with 64 3x3-filters this time
    # cnn_model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    # # MaxPooling layer
    # cnn_model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    # # flatten, to make inputs for the MLP neural net
    # cnn_model.add(Flatten())
    # # fully connected MLP, with 100 neurons in its first hidden layer
    # cnn_model.add(Dense(100, activation="relu"))
    # # dropout set to 50% of randomly selected neurons and their weights
    # cnn_model.add(Dropout(0.5))
    # # apply softmax to produce normalised output probabilities
    # cnn_model.add(Dense(10, activation="softmax"))  # there are 10 classes
    # cnn_model.summary()  # print the CNN model configuration to the console

    # cnn_model.compile(loss="categorical_crossentropy", optimizer="adam",
    #                   metrics=["accuracy"])
    # batch_size = 128  # set the batch size for updating weights
    # num_epochs = 5  # train the model for 5 epochs
    # # train the model, set "verbose=1" to show the training process
    # model_log = cnn_model.fit(train_x, train_y, batch_size=batch_size,
    #                           epochs=num_epochs, validation_data=(test_x, test_y), verbose=1)
    #
    # score = Model.evaluate(test_x, test_y, verbose=1)
    # print(f"Test loss: {score[0]}")
    # print(f"Test accuracy: {score[1]}")


def compare():
    # plot models accuracy against time taken
    pass


if __name__ == "__main__":
    # test()
    # lem_vs_stem("bow")

    # tfidf
    # lem_vs_stem("tfidf")
    # best_dataset("tfidf")
    # mlp_finetune("tokens_pos", "class_label", "tfidf") # 3164.secs

    # BoW
    # lem_vs_stem("bow")
    # best_dataset("bow")
    # mlp_finetune("tokens_pos", "class_label", "bow")


    cnn("tfidf", "tokens_pos", "class_label")
    # best_dataset("bow")

""""
  
        
        
        
    ************************
    
    reduction strategy:
        - for each vectorization type: 
            - for each dataset:
                - assess lem vs stem in cross validation -> best stays on == STEM
                - create lem or stem with  pos
                - cross val to get best dataset overall
            - fine tune with best dataset
            - save results and compare
                    
"""