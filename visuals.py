import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

def create_dataset_comp():
    pass


def create_mlp_graphs():
    # plot scatter between tfidf and bow fine-tunings
    v_type = ['tfidf', 'bow']
    for v in v_type:
        df = pd.read_csv(f"ft_results//mlp_ft_{v}_tokens_pos.csv")
        x = df['mean_fit_time'].to_numpy()
        y = df['mean_test_score'].to_numpy()

        plt.scatter(x, y, label=v)
    plt.title("TFIDF VS BoW MLP Fine-Tuning")
    plt.xlabel("Time")
    plt.ylabel("Mean Accuracy")
    plt.legend(["TFIDF", "BoW"])
    plt.savefig("graphs//mlp_ft_comparison.png")
    plt.show()

def create_cnn_graphs():
    pass

def compare_nn():
    pass

def add_labels(x, y, t1, t2):
    for i in range(len(x)):
        plt.text(x[i], y[i], f"{t1[i]},{t2[i]}", fontsize=6)

def create_topic_graphs():
    with open("nlu_data//model_results.json", "r") as file:
        model_rs = json.load(file)

    df = pd.DataFrame(model_rs['model_results'])
    print(df.head())
    print(df.columns)

    v_type = ['tfidf', 'bow']
    for v in v_type:
        df_v = df[df["v_type"] == v]
        x = df_v['avg_similarity'].to_numpy()
        y = df_v['total_time'].to_numpy()
        n_comp = df_v['n_components'].to_numpy()
        max_it = df_v['max_iter'].to_numpy()
        plt.scatter(x, y, label=v)
        add_labels(x, y, n_comp, max_it)


    plt.title("TFIDF VS BoW Topic Cosine Similarity")
    plt.ylabel("Time")
    plt.xlabel("Mean Similarity")
    plt.legend(["TFIDF", "BoW"])
    # plt.savefig("graphs//lda_v_similarity.png")
    plt.show()

    # compare best model to headline truths
    # best model = bow, n_comp=10, max_iter=50
    df_bm = pd.read_csv("nlu_data//bow_sim_2_10.csv")
    df_posts = pd.read_csv("nlu_data//tokens_lem.csv")

    combined = pd.merge(df_bm, df_posts, on=["id"], how="inner")
    print(combined.head())

    combined_true = combined[combined["class_label"] == 1]['similarity']
    combined_false = combined[combined["class_label"] == 0]['similarity']

    plt.hist(combined_true, bins=50, alpha=0.5, label="True")
    plt.hist(combined_false, bins=50, alpha=0.5, label="True")
    plt.xlabel("Similarity")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Similarity Distribution")
    plt.savefig("graphs//lda_sim_distribution.png")
    plt.show()



def test():

    pass


if __name__ == "__main__":
    test()
    # create_mlp_graphs()
    create_topic_graphs()