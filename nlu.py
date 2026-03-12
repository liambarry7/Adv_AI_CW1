import pickle

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse import load_npz


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




def test():

    pass


if __name__ == "__main__":
    test()

    x = load_npz("nlu_data/tokens_lem_headline.npz")
    x2 = load_npz("nlu_data/tokens_lem_post.npz")
    print(type(x))
    # get_topics(x, 5)
    get_topics(x2, 5)


