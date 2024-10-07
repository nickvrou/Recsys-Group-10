import numpy as np
import re
import gensim
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained Word2Vec model (Google News, 300 dimensions)
# This model helps find similarities between text descriptions.
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
    "C:/Users/astratou/Downloads/GoogleNews-vectors-negative300.bin.gz", binary=True)


def filter_listings_by_constraints(df, max_price, min_price, min_accommodates):
    """
    Filters listings based on price and accommodates.
    """
    return df[(df['price'] >= min_price) & (df['price'] <= max_price) & (df['accommodates'] >= min_accommodates)]


def sentence_to_embedding(sentence, model, embedding_dim=300):
    """
    Converts a sentence into an average embedding vector using Word2Vec.
    """
    words = re.findall(r'\w+', sentence.lower())
    embedding_vectors = [model[word] for word in words if word in model]
    if len(embedding_vectors) == 0:
        return np.zeros(embedding_dim)
    return np.mean(embedding_vectors, axis=0)


def cosine_similarity_embeddings(embedding1, embedding2):
    """
    Calculates the cosine similarity between two embedding vectors.
    """
    return cosine_similarity([embedding1], [embedding2])[0][0]


def explain_recommendation_with_word2vec(user_description, listing_description, listing_comments, model,
                                         popularity_score=None):
    """
    Generates an explanation for why a listing is recommended based on Word2Vec similarity.
    """
    combined_listing_text = f"{listing_description} {listing_comments}"
    user_embedding = sentence_to_embedding(user_description, model)
    listing_embedding = sentence_to_embedding(combined_listing_text, model)

    # Get top matching keywords and phrases
    matching_keywords = extract_matching_keywords(user_description, combined_listing_text, model)
    explanation = f"Recommended due to key phrase matches: {', '.join(matching_keywords)}."

    # Include similarity score and popularity score (if available)
    similarity_score = cosine_similarity_embeddings(user_embedding, listing_embedding)
    explanation += f" Similarity score: {similarity_score:.2f}."
    if popularity_score is not None:
        explanation += f" Popularity score: {popularity_score:.2f}."

    return explanation


def extract_matching_keywords(user_text, listing_text, model):
    """
    Finds common keywords between user input and listing text.
    """
    user_words = re.findall(r'\w+', user_text.lower())
    listing_words = re.findall(r'\w+', listing_text.lower())
    return [word for word in listing_words if word in user_words and word in model][:5]


def remove_stop_words(text, stop_words):
    """
    Cleans the text by removing stop words and punctuation.
    """
    if pd.isnull(text):
        return text
    words = re.findall(r'\w+', text)
    return ' '.join([word for word in words if word.lower() not in stop_words])
