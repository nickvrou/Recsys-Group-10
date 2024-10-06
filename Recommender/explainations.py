import numpy as np
import re
import gensim
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained Word2Vec model (Google News, 300 dimensions)
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format("C:/Users/astratou\Downloads\GoogleNews-vectors-negative300.bin.gz", binary=True)

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

def explain_recommendation_with_word2vec(user_description, listing_description, listing_comments, model, popularity_score=None):
    """
    Explain why the listing is recommended based on semantic similarity using Word2Vec.
    Includes the key phrases and popularity score (if available), without the similarity score.
    """
    # Combine the listing description and comments for comparison
    combined_listing_text = f"{listing_description} {listing_comments}"

    # Convert the user description and listing text to embeddings
    user_embedding = sentence_to_embedding(user_description, model)
    listing_embedding = sentence_to_embedding(combined_listing_text, model)

    # Get keywords that match between the user input and listing descriptions
    user_keywords = extract_keywords(user_description, model)
    listing_keywords = extract_keywords(combined_listing_text, model)
    listing_phrases = extract_phrases(combined_listing_text, model)


    # Generate explanation based on matched keywords
    explanation = f"This listing is recommended based on key phrases such as: {', '.join(listing_keywords)}."

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
    # Add logic here to detect common phrases (like 'free Wi-Fi', 'close to public transport')
    words = re.findall(r'\w+', sentence.lower())
    phrases = []
    for i in range(len(words) - 1):
        phrase = f"{words[i]} {words[i + 1]}"
        if phrase in model:
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
