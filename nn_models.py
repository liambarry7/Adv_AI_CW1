from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
import pandas as pd
from scipy.sparse import load_npz

import time

def mlp(target_csv, column):

    start = time.time()

    # mlp = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=1000, early_stopping=True)
    mlp = MLPClassifier()

    training_set = pd.read_csv(f"{target_csv}_training.csv")
    test_set = pd.read_csv(f"{target_csv}_test.csv")

    # training_set = pd.read_csv("bow//post-tokens_lem_training.csv")
    # test_set = pd.read_csv("bow//post-tokens_lem_test.csv")

    train_y = training_set[column]
    test_y = test_set[column]

    # train_y = training_set['class_label']
    # test_y = test_set['class_label']

    train_x = load_npz(f"{target_csv}_train.npz")
    test_x = load_npz(f"{target_csv}_test.npz")

    # train_x = load_npz("bow//post-tokens_lem_train.npz")
    # test_x = load_npz("bow//post-tokens_lem_test.npz")

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


def mlp_finetune(folder, target_csv, column):

    start = time.time()


    # get datasets
    training_set = pd.read_csv(f"{folder}{target_csv}_training.csv")
    test_set = pd.read_csv(f"{folder}{target_csv}_test.csv")

    train_y = training_set[column]
    test_y = test_set[column]

    train_x = load_npz(f"{folder}{target_csv}_train.npz")
    test_x = load_npz(f"{folder}{target_csv}_test.npz")

    # create a new mlp()
    mlp = MLPClassifier(max_iter=1000, early_stopping=True)

    # define 3 hyperparams to fine-tune
    mlp_hyperparams = [{
        'hidden_layer_sizes': [(100, 50), (64,32), (100,)],
        'activation': ['relu', 'logistic', 'tanh'],
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
        ['param_activation', 'param_hidden_layer_sizes', 'param_learning_rate', 'mean_test_score',
         'std_test_score', 'rank_test_score']].sort_values(by='rank_test_score')
    print(results_df.head())

    results_df.to_csv(f'ft_results//mlp_ft_{target_csv}.csv', index=False)



def test():

    pass


if __name__ == "__main__":
    # test()
    # mlp("bow//post-tokens_lem", "class_label")
    mlp_finetune("bow//", "post-tokens_lem", "class_label")
