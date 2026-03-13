import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse import save_npz, load_npz, vstack
from sklearn.metrics.pairwise import cosine_similarity
import time
import json

def topics_extraction(v_type):
    n_comps = [2, 5, 7, 10]
    max_iters = [10, 25, 50]

    for n in n_comps:
        for i in max_iters:
            get_topics(v_type, n, i)


def get_topics(v_type, n_comp, max_iterations):
    print("Fitting LDA model...")
    # get datasets
    headlines = load_npz(f"nlu_data/tokens_lem_{v_type}_headline.npz")
    posts = load_npz(f"nlu_data/tokens_lem_{v_type}_post.npz")

    full_data = vstack([headlines, posts])

    # train/ fit_transform the vectoriser on all the data
    lda_model = LatentDirichletAllocation(n_components=n_comp, learning_method='online', random_state=42, max_iter=max_iterations)
    params = lda_model.get_params()

    start = time.time()

    # fit the lda model with the full corpus
    lda_model.fit(full_data)

    # get topics from headlines and posts
    print("Transforming headlines and posts...")
    headlines_topics = lda_model.transform(headlines)
    posts_topics = lda_model.transform(posts)

    end = time.time()

    # save topics
    np.save(f"nlu_data//{v_type}_headlines_{params['n_components']}_{params['max_iter']}.npy", headlines_topics)
    np.save(f"nlu_data//{v_type}_post_{params['n_components']}_{params['max_iter']}.npy", posts_topics)

    # save model
    with open(f'nlu_data//lda_{v_type}_{params['n_components']}_{params['max_iter']}_m.pkl', 'wb') as file:
        pickle.dump(lda_model, file)

    # print topic-word distribution
    print(lda_model.components_)
    print(lda_model.components_.shape) #(n_topics, vocabulary size)

    # print_top_words(lda_model, "x", 0, 10)

    print("Calculating headline-post topic similarity...")
    # compare using cosine similarity
    similarities = np.array([cosine_similarity(headlines_topics[i].reshape(1,-1), posts_topics[i].reshape(1,-1))[0][0] for i in range (headlines_topics.shape[0])])
    print(similarities.shape)
    print(similarities[:500])

    avg_similarity = np.mean(similarities)
    total_time = end - start
    print(f"average similarity: {avg_similarity}")
    print(f"min similarity: {np.min(similarities)}")
    print(f"max similarity: {np.max(similarities)}")
    print(f"median similarity: {np.median(similarities)}")
    print(f"time taken: {np.median(total_time)}")


    print(headlines_topics[:5])
    print(posts_topics[:5])

    # save
    df_similarity = pd.DataFrame({
        'id' : np.arange(len(similarities))+1,
        'similarity': similarities
    })

    df_similarity.to_csv(f"nlu_data//{v_type}_sim_{params['n_components']}_{params['max_iter']}.csv", index=False)


    model_data = {
        "v_type" : v_type,
        "n_components" : params['n_components'],
        "max_iter" : params['max_iter'],
        "avg_similarity" : avg_similarity,
        "min_similarity" : np.min(similarities),
        "max_similarity" : np.max(similarities),
        "median_similarity" : np.median(similarities),
        "total_time" : total_time
    }

    with open("nlu_data//model_results.json", "r") as file:
        model_rs = json.load(file)

    model_rs['model_results'].append(model_data)

    with open("nlu_data//model_results.json", "w") as file:
        json.dump(model_rs, file, indent=4)


if __name__ == "__main__":

    topics_extraction("tfidf")
    topics_extraction("bow")



