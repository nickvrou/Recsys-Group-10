import numpy as np
import re
import gensim
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# Load the pre-trained Word2Vec model (Google News, 300 dimensions)
# This model allows for semantic similarity comparisons based on word embeddings.
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
    "C:/Users/astratou/Downloads/GoogleNews-vectors-negative300.bin.gz", binary=True)


def filter_listings_by_constraints(df, max_price, min_price, min_accommodates):
    """
    Filters the Airbnb listings based on price range and number of people it accommodates.

    Args:
    df (DataFrame): The listings dataset.
    max_price (float): The maximum price for filtering.
    min_price (float): The minimum price for filtering.
    min_accommodates (int): Minimum number of people the listing should accommodate.

    Returns:
    DataFrame: Filtered DataFrame with listings that meet the price and accommodation constraints.
    """
    filtered_df = df[(df['price'] >= min_price) & (df['price'] <= max_price) & (df['accommodates'] >= min_accommodates)]
    return filtered_df


def sentence_to_embedding(sentence, model, embedding_dim=300):
    """
    Converts a sentence into a vector representation by averaging the word embeddings from Word2Vec.

    Args:
    sentence (str): The input sentence.
    model (gensim Word2Vec model): Pre-trained Word2Vec model.
    embedding_dim (int): Dimension of the word embeddings.

    Returns:
    np.array: Average embedding vector for the sentence.
    """
    words = re.findall(r'\w+', sentence.lower())
    embedding_vectors = [model[word] for word in words if word in model]

    if len(embedding_vectors) == 0:
        # Return a zero vector if none of the words are found in the model
        return np.zeros(embedding_dim)

    return np.mean(embedding_vectors, axis=0)


def cosine_similarity_embeddings(embedding1, embedding2):
    """
    Computes the cosine similarity between two embedding vectors.

    Args:
    embedding1 (np.array): The first embedding.
    embedding2 (np.array): The second embedding.

    Returns:
    float: Cosine similarity between the two embeddings.
    """
    return cosine_similarity([embedding1], [embedding2])[0][0]


def explain_recommendation_with_word2vec(user_description, listing_description, listing_comments, model, popularity_score=None):
    """
    Generates an explanation for why a listing is recommended based on Word2Vec semantic similarity.

    Args:
    user_description (str): Description provided by the user.
    listing_description (str): Description of the recommended listing.
    listing_comments (str): Comments for the recommended listing.
    model (gensim Word2Vec model): Pre-trained Word2Vec model.
    popularity_score (float, optional): Popularity score for the listing.

    Returns:
    str: An explanation of the recommendation, mentioning key phrases and similarity score.
    """
    # Combine the listing description and comments for comparison
    combined_listing_text = f"{listing_description} {listing_comments}"

    # Convert the user input and listing text to embedding vectors
    user_embedding = sentence_to_embedding(user_description, model)
    listing_embedding = sentence_to_embedding(combined_listing_text, model)

    # Extract keywords and phrases for more detailed explanation
    user_keywords = extract_keywords(user_description, model)
    listing_keywords = extract_keywords(combined_listing_text, model)
    listing_phrases = extract_phrases(combined_listing_text, model)

    # Calculate similarity between the user's input and the listing
    similarity_score = cosine_similarity_embeddings(user_embedding, listing_embedding)

    # Find matching keywords between user input and listing
    matching_keywords = [word for word in listing_keywords if word in user_keywords]

    # Generate the explanation text
    explanation = f"This listing is recommended because it matches your input with key phrases like: {', '.join(matching_keywords)}."

    if listing_phrases:
        explanation += f" The listing also highlights phrases such as: {', '.join(listing_phrases)}."

    explanation += f" The similarity score between your description and this listing is {similarity_score:.2f}."

    if popularity_score is not None:
        explanation += f" The popularity score for this listing is {popularity_score:.2f}."

    return explanation


def extract_keywords(sentence, model):
    """
    Extracts important words from the sentence that have corresponding embeddings in the Word2Vec model.

    Args:
    sentence (str): Input sentence.
    model (gensim Word2Vec model): Pre-trained Word2Vec model.

    Returns:
    list: List of top 5 extracted keywords.
    """
    words = re.findall(r'\w+', sentence.lower())
    keywords = [word for word in words if word in model]
    return keywords[:5]  # Return the top 5 words for explanation


def extract_phrases(sentence, model):
    """
    Extracts common two-word phrases from the sentence based on Word2Vec embeddings.

    Args:
    sentence (str): Input sentence.
    model (gensim Word2Vec model): Pre-trained Word2Vec model.

    Returns:
    list: List of top 5 extracted phrases.
    """
    words = re.findall(r'\w+', sentence.lower())
    phrases = []
    for i in range(len(words) - 1):
        phrase = f"{words[i]} {words[i + 1]}"
        if all(word in model for word in phrase.split()):
            phrases.append(phrase)
    return phrases[:5]  # Return the top 5 phrases


def remove_stop_words(text, stop_words):
    """
    Removes stop words and punctuation from the given text.

    Args:
    text (str): Input text.
    stop_words (list): List of stop words to remove.

    Returns:
    str: Cleaned text with stop words and punctuation removed.
    """
    if pd.isnull(text):
        return text

    # Remove punctuation and non-alphabetical characters
    text = re.sub(r'[^\w\s]', '', text)

    # Split text into individual words
    words = text.split()

    # Filter out stop words
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Rejoin the cleaned words back into a single sentence
    filtered_text = ' '.join(filtered_words)

    return filtered_text