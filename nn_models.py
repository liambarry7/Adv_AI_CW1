from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
import pandas as pd
from scipy.sparse import load_npz, hstack

import time

def mlp(target_csv, column):

    start = time.time()

    # mlp = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=1000, early_stopping=True)
    mlp = MLPClassifier(max_iter=1000, early_stopping=True)

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


def mlp_finetune(target_csv, column, vectorizerType):

    start = time.time()


    # get datasets
    training_set = pd.read_csv(f"{vectorizerType}//{target_csv}_training.csv")
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

def best_dataset(v_type, datasets):
    # use cross validation to select the best dataset
    datasets = ["post-tokens_lem", "post-tokens_stem", "post-entities", "post-pos", "lem_ner", "lem_pos", "lem_ner_pos",
                "stem_ner", "stem_pos", "stem_ner_pos"]
    results = []

    for d in datasets:
        print(f"\nTesting {d}...")
        training_set = pd.read_csv(f"{v_type}//{d}_training.csv")
        # test_set = pd.read_csv(f"{v_type}//{d}_test.csv")

        train_y = training_set["class_label"]
        # test_y = test_set["class_label"]

        train_x = load_npz(f"{v_type}//{d}_train.npz")
        # test_x = load_npz(f"{v_type}//{d}_test.npz")

        mlp = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=1000, early_stopping=True)

        ff = StratifiedKFold(n_splits=5, shuffle=True, random_state=41)  # use StratifiedKFold to avoid imbalanced class distribution

        scores = cross_val_score(mlp, train_x, train_y, cv=ff, n_jobs=-1)

        mean_cv = scores.mean()
        std_cv = scores.std()
        print(f"CV accuracy scores: {scores}")
        print(f"Mean CV accuracy: {mean_cv}")
        print(f"Standard deviation: {std_cv}")

        rs = {"v_type" : v_type,
               "name" : d,
               "mean_cv_accuracy" : mean_cv,
               "std" : std_cv}

        results.append(rs)


    df = pd.DataFrame(results)
    df.to_csv("dataset_comparison_test.csv")


def test():

    pass


if __name__ == "__main__":
    # test()
    # lem_vs_stem("bow")
    lem_vs_stem("tfidf")

    # mlp("bow//post-tokens_lem", "class_label")
    # mlp("bow//post-tokens_stem", "class_label")

    # fine tune tokens
    # mlp_finetune("post-tokens_lem", "class_label", "bow") # done
    # mlp_finetune("post-tokens_stem", "class_label", "bow") # done

    # mlp_finetune("post-tokens_lem", "class_label", "tfidf") # done
    # mlp_finetune("post-tokens_stem", "class_label", "tfidf") # done

    # fine tune entities
    # mlp_finetune("post-entities", "class_label", "bow")
    # mlp_finetune("post-entities", "class_label", "tfidf")

    # fine tune pos
    # mlp_finetune("post-pos", "class_label", "bow")
    # mlp_finetune("post-pos", "class_label", "tfidf")

    # fine tune lem entities
    # mlp_finetune("lem_ner_pos", "class_label", "bow")



    # best_dataset("bow")

""""
    Testing plan:
        - for each v_type:
            - assess lem vs stem -> winner goes all the way
            - then winner + pos, winner + ner
            - then all 3 together (and vs loser all 3 if time)
            - total = 2 + 2 + 1 (+1?) = 5 runs per v_type
        
            - create graphs to compare all runnings
        
        - compare best combinations for all v_types and params
        
        - repeat for cnn    
        
        
        
    ************************
    
    reduction strategy:
        - for each vectorization type: - wrong, only need one, this is about the classifiers
            - for each dataset:
                - assess lem vs stem in cross validation -> best stays on == STEM
                - create lem or stem with  pos
                - cross val to get best dataset overall
            - fine tune with best dataset
            - save results and compare
                    
"""