from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
import pandas as pd
from scipy.sparse import load_npz
import pickle

import tensorflow as tf
# print(f"tensorflow:{tf.__version__}")

from keras.layers import Conv1D, Dropout, Dense, GlobalMaxPooling1D
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import json
import time

def mlp(v_type, target_csv, column):
    # relu,"(64, 32)",adaptive
    mlp = MLPClassifier(activation= 'relu', hidden_layer_sizes=(64,32), learning_rate='adaptive', max_iter=1000, early_stopping=True)

    training_set = pd.read_csv(f"{v_type}//{target_csv}_train.csv")
    train_y = training_set[column]

    train_x = load_npz(f"{v_type}//{target_csv}_train.npz")

    mlp.fit(train_x, train_y)

    # save model
    with open(f'best_{v_type}_mlp.pkl', 'wb') as file:
        pickle.dump(mlp, file)

def test_mlp(vectorizerType, target_csv, column):
    # get datasets
    training_set = pd.read_csv(f"{vectorizerType}//{target_csv}_train.csv")
    test_set = pd.read_csv(f"{vectorizerType}//{target_csv}_test.csv")

    train_y = training_set[column]
    test_y = test_set[column]

    train_x = load_npz(f"{vectorizerType}//{target_csv}_train.npz")
    test_x = load_npz(f"{vectorizerType}//{target_csv}_test.npz")

    # relu,"(64, 32)",adaptive
    mlp = MLPClassifier(activation='relu', hidden_layer_sizes=(64, 32), learning_rate='adaptive', max_iter=1000,
                        early_stopping=True)

    mlp.fit(train_x, train_y)

    y_predictions = mlp.predict(test_x)

    print(y_predictions)
    print(f"Accuracy: {accuracy_score(test_y, y_predictions)}") # Accuracy: 0.9217182973033207



def mlp_finetune(target_csv, column, vectorizerType):

    start = time.time()

    # get datasets
    training_set = pd.read_csv(f"{vectorizerType}//{target_csv}_train.csv")
    test_set = pd.read_csv(f"{vectorizerType}//{target_csv}_test.csv")

    train_y = training_set[column]

    train_x = load_npz(f"{vectorizerType}//{target_csv}_train.npz")

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

def get_cnn_model(filters, kernel_size, max_len, learning_rate):
    # input_dim = no of unique tokens
    # output_dim = size of word vector

    # filters = no of feature detectors
    # kernel_size = no of words looked at at once

    # instantiate a CNN model, Sequential type
    cnn_model = Sequential([
        Conv1D(filters=filters, kernel_size=kernel_size, input_shape=(max_len, 1), activation='relu'),
        GlobalMaxPooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    cnn_model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='binary_crossentropy', metrics=['accuracy'])

    return cnn_model

def fine_tune_cnn(vectorizerType, target_csv, column):
    # get datasets
    training_set = pd.read_csv(f"{vectorizerType}/{target_csv}_train.csv")
    test_set = pd.read_csv(f"{vectorizerType}/{target_csv}_test.csv")

    train_y = training_set[column]
    test_y = test_set[column]

    train_x = load_npz(f"{vectorizerType}/{target_csv}_train.npz").toarray()
    test_x = load_npz(f"{vectorizerType}/{target_csv}_test.npz").toarray()

    print(test_x.shape)

    # set max_len param
    max_len = train_x.shape[1]

    # expects 3D tensor = (Batch size, sequence length, embedding dimensions)
    train_x = train_x.reshape(train_x.shape[0], max_len, 1)
    test_x = test_x.reshape(test_x.shape[0], max_len, 1)

    filters = [128, 64]
    kernel_sizes = [3, 5]
    learning_rates = [0.001, 0.0001]

    for f in filters:
        for k in kernel_sizes:
            for l in learning_rates:
                cnn_model = get_cnn_model(f, k, max_len, l)
                start = time.time()

                # train the model, set "verbose=1" to show the training process
                model_log = cnn_model.fit(train_x, train_y, batch_size=128,
                                          epochs=5, validation_data=(test_x, test_y),
                                          callbacks=EarlyStopping(monitor='val_loss',
                                                                  patience = 2,
                                                                  restore_best_weights=True),
                                          verbose=1)
                end = time.time()

                score = cnn_model.evaluate(test_x, test_y, verbose=1)
                print(f"Test loss: {score[0]}")
                print(f"Test accuracy: {score[1]}")

                total_time = end - start

                model_data = {
                    "v_type": vectorizerType,
                    "filter": f,
                    "kernel_size": k,
                    "learning_rate": l,
                    "test_loss": score[0],
                    "test_acc": score[1],
                    "total_time": total_time
                }

                with open(f"ft_results//cnn_ft.json", "r") as file:
                    model_rs = json.load(file)

                model_rs['model_results'].append(model_data)

                with open(f"ft_results//cnn_ft.json", "w") as file:
                    json.dump(model_rs, file, indent=4)

def cnn(vectorizerType, target_csv, column):
    training_set = pd.read_csv(f"{vectorizerType}/{target_csv}_train.csv")
    test_set = pd.read_csv(f"{vectorizerType}/{target_csv}_test.csv")

    train_y = training_set[column]
    test_y = test_set[column]

    train_x = load_npz(f"{vectorizerType}/{target_csv}_train.npz").toarray()
    test_x = load_npz(f"{vectorizerType}/{target_csv}_test.npz").toarray()

    print(test_x.shape)

    max_len = train_x.shape[1]

    # expects 3D tensor = (Batch size, sequence length, embedding dimensions)
    train_x = train_x.reshape(train_x.shape[0], max_len, 1)
    test_x = test_x.reshape(test_x.shape[0], max_len, 1)

    # input_dim = no of unique tokens
    # output_dim = size of word vector

    # filters = no of feature detectors
    # kernel_size = no of words looked at at once

    # instantiate a CNN model, Sequential type
    cnn_model = Sequential([
        Conv1D(filters=64, kernel_size=3, input_shape=(max_len, 1), activation='relu'),
        GlobalMaxPooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # save model
    with open('best_tfidf_cnn.pkl', 'wb') as file:
        pickle.dump(cnn_model, file)

    # fine tune: filters, kernel_size, dropout_rate

    batch_size = 128  # set the batch size for updating weights
    num_epochs = 5  # train the model for 5 epochs
    # train the model, set "verbose=1" to show the training process
    model_log = cnn_model.fit(train_x, train_y, batch_size=batch_size,
                              epochs=num_epochs, validation_data=(test_x, test_y), verbose=1)

    score = cnn_model.evaluate(test_x, test_y, verbose=1)
    print(f"Test loss: {score[0]}")
    print(f"Test accuracy: {score[1]}")
#     Test loss: 0.6806715726852417
# Test accuracy: 0.5783931612968445

    # save trained model
    # save model
    with open('best_bow_cnn.pkl', 'wb') as file:
        pickle.dump(mlp, file)


if __name__ == "__main__":
    lem_vs_stem("bow")

    # tfidf
    lem_vs_stem("tfidf")
    best_dataset("tfidf")
    mlp_finetune("tokens_pos", "class_label", "tfidf") # 3164.secs

    # BoW
    lem_vs_stem("bow")
    best_dataset("bow")
    mlp_finetune("tokens_pos", "class_label", "bow")

    # train and save the best model
    mlp("bow", "tokens_pos", "class_label")
    mlp("tfidf", "tokens_pos", "class_label")
    test_mlp("bow", "tokens_pos", "class_label")


    cnn("tfidf", "tokens_pos", "class_label")
    fine_tune_cnn("tfidf", "tokens_pos", "class_label")
    fine_tune_cnn("bow", "tokens_pos", "class_label")

