import numpy as np
import re
import gensim
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# Load the pre-trained Word2Vec model (Google News, 300 dimensions)
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
    'recommender\GoogleNews-vectors-negative300.bin.gz', binary=True)


def filter_listings_by_constraints(df, max_price, min_price, min_accommodates):
    """
    Filter listings based on price and the number of people it accommodates.
    """
    filtered_df = df[(df['price'] >= min_price) & (df['price'] <= max_price) & (df['accommodates'] >= min_accommodates)]
    return filtered_df


def sentence_to_embedding(sentence, model, embedding_dim=300):
    """
    Convert a sentence to an embedding by averaging Word2Vec embeddings of the words.
    """
    words = re.findall(r'\w+', sentence.lower())
    embedding_vectors = [model[word] for word in words if word in model]

    if len(embedding_vectors) == 0:
        # Return a zero vector if no words from the sentence are in the model
        return np.zeros(embedding_dim)

    return np.mean(embedding_vectors, axis=0)


def cosine_similarity_embeddings(embedding1, embedding2):
    """Compute the cosine similarity between two embeddings."""
    return cosine_similarity([embedding1], [embedding2])[0][0]


def explain_recommendation_with_word2vec(user_description, listing_description, listing_comments, model):
    """
    Explain why the listing is recommended based on semantic similarity using Word2Vec.
    Includes key phrases and compares them with the user's input to provide better context.
    """
    # Combine the listing description and comments for comparison
    combined_listing_text = f"{listing_description} {listing_comments}"

    # Convert the user description and listing text to embeddings
    user_embedding = sentence_to_embedding(user_description, model)
    listing_embedding = sentence_to_embedding(combined_listing_text, model)

    # Get keywords and phrases for both user and listing
    user_keywords = extract_keywords(user_description, model)
    listing_keywords = extract_keywords(combined_listing_text, model)
    listing_phrases = extract_phrases(combined_listing_text, model)

    # Calculate similarity score
    similarity_score = cosine_similarity_embeddings(user_embedding, listing_embedding)

    # Find matching keywords between the user description and listing
    matching_keywords = [word for word in listing_keywords if word in user_keywords]

    # Generate explanation based on matched keywords and phrases
    explanation = f"This listing is recommended because it matches your input with key phrases like: {', '.join(matching_keywords)}."

    if listing_phrases:
        explanation += f" The listing also highlights phrases such as: {', '.join(listing_phrases)}."

    # Add similarity score to the explanation (optional, can be used for further context)
    explanation += f" The similarity score between your description and this listing is {similarity_score:.2f}."

    return explanation


def extract_keywords(sentence, model):
    """
    Extract words from the sentence that have embeddings in the Word2Vec model.
    """
    words = re.findall(r'\w+', sentence.lower())
    keywords = [word for word in words if word in model]
    return keywords[:5]  # Return top 5 words for explanation


def extract_phrases(sentence, model):
    """
    Extract common phrases from the sentence based on Word2Vec embeddings.
    This can be useful for identifying common amenities or specific key phrases in descriptions.
    """
    words = re.findall(r'\w+', sentence.lower())
    phrases = []
    for i in range(len(words) - 1):
        phrase = f"{words[i]} {words[i + 1]}"
        if all(word in model for word in phrase.split()):
            phrases.append(phrase)
    return phrases[:5]  # Return top 5 phrases


def remove_stop_words(text, stop_words):
    """
    Remove stop words from the text, and also clean up punctuation.
    """
    if pd.isnull(text):
        return text

    # Remove punctuation and non-alphabetical characters
    text = re.sub(r'[^\w\s]', '', text)

    # Split text into words
    words = text.split()

    # Filter out stop words
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Rejoin the filtered words into a cleaned sentence
    filtered_text = ' '.join(filtered_words)

    return filtered_text
