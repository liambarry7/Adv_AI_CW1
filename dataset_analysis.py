import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

def preliminary_descriptive_analysis():
    df = pd.read_csv("raw_datasets//social-media-release.csv")
    print(df.head())

    print(f"Columns: {df.columns}")
    # ['id', 'news_headline', 'news_headline_ground_truth', 'post', 'majority_votes', 'class_label']
    print("*"*50)

    example_row = df.iloc[0]
    # print(example_row)

    print(f"ID: {example_row.id}\n",
          f"headline: {example_row.news_headline}\n",
          f"post: {example_row.post}\n",
          f"majority votes: {example_row.majority_votes}\n",
          f"ground truth: {example_row.news_headline_ground_truth}\n",
          f"class label: {example_row.class_label}")

    print("*"*50)

    # get total no of posts
    total_posts = df.shape[0]
    print(f"total posts: {total_posts}")
    print("*"*50)

    # count ground truth t/f values
    no_of_ground_truth = df['news_headline_ground_truth'].value_counts()
    print(f"no ground truth:\n{no_of_ground_truth}")
    ground_truth_perc = (no_of_ground_truth.iloc[0] / total_posts) * 100
    print(f"% ground truth = True: {ground_truth_perc}")
    print("*"*50)

    # count class truth (label) t/f values
    no_of_class_truth = df['class_label'].value_counts()
    print(f"no class truth:\n{no_of_class_truth}")
    class_truth_perc = (no_of_class_truth.iloc[0] / total_posts) * 100
    print(f"% class truth = True {class_truth_perc}")
    print("*"*50)

    # count majority votes (agree/disagree) values
    no_of_majority_votes = df['majority_votes'].value_counts()
    print(f"no majority votes:\n{no_of_majority_votes}")
    no_of_majority_perc = (no_of_majority_votes.iloc[0] / total_posts) * 100
    print(f"% Majority votes = True {no_of_majority_perc}")
    print("*" * 50)

    # remap majority values to be true/ false to match other columns
    df['majority_votes_tf'] = df['majority_votes'].map({
        "Agree": True,
        "Disagree": False
    })

    # count tp, tn, fp, fn
    tp = len(df[(df['news_headline_ground_truth'] == True) & (df['majority_votes_tf'] == True)])
    tn = len(df[(df['news_headline_ground_truth'] == False) & (df['majority_votes_tf'] == False)])
    fp = len(df[(df['news_headline_ground_truth'] == False) & (df['majority_votes_tf'] == True)])
    fn = len(df[(df['news_headline_ground_truth'] == True) & (df['majority_votes_tf'] == False)])

    print("TP:", tp)
    print("TN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("*" * 50)

    # bar chart for each binary t/f column (label, majority, ground truths) & cm
    fig, axs = plt.subplots(2,2)
    colours = ['green', 'red']
    x = ['True', 'False']
    x_alternate = ['Agree', 'Disagree']
    yticks = range(35000, 60000, 5000)

    ax1 = axs[0,0]
    y1 = [no_of_ground_truth.iloc[0], no_of_ground_truth.iloc[1]]
    ax1.bar(x, y1, color=colours)
    ax1.set_title("Ground Truths")
    ax1.set_ylabel("Count")
    ax1.set_xlabel("Boolean Value")
    ax1.set_yticks(yticks)
    ax1.set_ylim(35000, 60000)

    ax2 = axs[0,1]
    y2 = [no_of_class_truth.iloc[0], no_of_class_truth.iloc[1]]
    ax2.bar(x, y2, color=colours)
    ax2.set_title("Class Truth")
    ax2.set_ylabel("Count")
    ax2.set_xlabel("Boolean Value")
    ax2.set_yticks(yticks)
    ax2.set_ylim(35000, 60000)

    ax3 = axs[1, 0]
    y2 = [no_of_majority_votes.iloc[0], no_of_majority_votes.iloc[1]]
    ax3.bar(x_alternate, y2, color=colours)
    ax3.set_title("Majority Votes")
    ax3.set_ylabel("Count")
    ax3.set_xlabel("Boolean Value")

    ax4 = axs[1,1]
    mlp_cm = metrics.ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix(df['news_headline_ground_truth'], df['majority_votes_tf']), display_labels=["False", "True"])
    mlp_cm.plot(ax=ax4, values_format='d')
    ax4.set_title("Majority Vote vs Ground Truth Matrix")
    ax4.set_xlabel("Majority Votes")
    ax4.set_ylabel("Ground Truths")

    plt.tight_layout(pad=2)
    plt.savefig("graphs/descriptive_anaylsis_stats.png")
    plt.show()


def test():
    print("Test harness")
    preliminary_descriptive_analysis()


if __name__ == "__main__":
    # test()
    preliminary_descriptive_analysis()