import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse import load_npz, vstack
from sklearn.metrics.pairwise import cosine_similarity
import time
import json


"""
    - use NLU & NLP to explore and analyse content in docuemtns and their linked news headlines
    to automatically discover topics across text data
    - implement standard syntactic analysis, including parsing/tokenisation, stop word removal,
    lemmatization and stemming.
    - experiment with at least two text representation strategies (e.g. LDA, BoW, TFIDF,
    word vector/embeddings
    - evaluate results from models, discuss discovered topics and their link to document content,
    news headlines and truth labels
    - save the best model


    - tokenise as per nlp
    - then do topic extraction from tokens, create a document-topic matrix
    - look at week 4 lab
    -> vectorise the tokens, then put into LDA for topic extraction
    -> compare the effectiveness of different vectorisation techniques

    - get intent from sentences?
    - use synonym lists?
    
    
    - get headlines topics, get post topics
    - see link between headlines topics and all of their posts' topics
    - link to both headlines and post class labels


"""





def get_topics(feature_vector, n_topics):
    # extract topic from tokens
    lda_model = LatentDirichletAllocation(n_components=n_topics, learning_method='online', random_state=42, max_iter=10)
    lda_model.fit(feature_vector)  # train the model

    # print the topic probability distribution for each document
    lda_top = lda_model.fit_transform(feature_vector)
    print(lda_top.shape)
    print(lda_top)

    # print topic-word distribution
    print(lda_model.components_)
    print(lda_model.components_.shape)

    # assess LDA model
    # perplexity
    perplexity = lda_model.perplexity(feature_vector)
    print(f"Perplexity: {perplexity}")

    # topic correlation
    import pandas as pd
    doc_topic_distribution = lda_model.transform(feature_vector)
    print(doc_topic_distribution.shape)
    topic_correlation = pd.DataFrame(doc_topic_distribution).corr()
    print(topic_correlation)

    # with open(f'{folder}//{target_csv}_tfidf_v_pt.pkl', 'wb') as file:
    with open(f'nlu_data//tokens_lem_tfidf_v_pt.pkl', 'rb') as file:
        vect = pickle.load(file)

    print_top_words(lda_model, vect, 0, 10)


def print_top_words(lda_model, vect, topic_index, n_top_words):
    with open(f'nlu_data//tokens_lem_tfidf_v_pt.pkl', 'rb') as file:
        vect = pickle.load(file)

    # with open(f'nlu_data//{vect}.pkl', 'rb') as file:
    #     vect = pickle.load(file)

    topic_word_mat = lda_model.components_[topic_index]
    vocabulary = vect.get_feature_names_out()
    vocab_comp = zip(vocabulary, topic_word_mat)

    # words are sorted by weight in reverse order, and slice the first n_top_words elements
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:n_top_words]

    print(f"Topic {topic_index}:")
    for word in sorted_words:
        print(word[0], end=" ")
    print("\n")


def topics_test(v_type):
    # get datasets
    headlines = load_npz("nlu_data/tokens_lem_tfidf_headline.npz")
    posts = load_npz("nlu_data/tokens_lem_tfidf_post.npz")


    full_data = vstack([headlines, posts])

    # train/ fit_transform the vectoriser on all the data
    lda_model = LatentDirichletAllocation(n_components=10, learning_method='online', random_state=42, max_iter=50)

    start = time.time()

    # fit the lda model with the full corpus
    lda_model.fit(full_data)

    # get topics from headlines and posts
    headlines_topics = lda_model.transform(headlines)
    posts_topics = lda_model.transform(posts)

    end = time.time()

    # print topic-word distribution
    print(lda_model.components_)
    print(lda_model.components_.shape) #(n_topics, vocabulary size)

    print_top_words(lda_model, "x", 0, 10)
    # print_topics(lda_model, )

    # save model and topics

    # split to new function here

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
    params = lda_model.get_params()
    v_type = "tfidf"

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


def test():

    pass


if __name__ == "__main__":
    test()

    x = load_npz("nlu_data/tokens_lem_headline.npz")
    x2 = load_npz("nlu_data/tokens_lem_post.npz")
    print(type(x))
    # get_topics(x, 5)
    # get_topics(x2, 5)

    topics_test()

