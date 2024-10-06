import numpy as np
import re
import gensim
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load the pre-trained Word2Vec model (Google News, 300 dimensions)
word2vec_path = 'C:/Users/astratou/Downloads/GoogleNews-vectors-negative300.bin.gz'
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)


# Helper function to get the index of a listing based on its ID
def get_listing_index(listing_id, df_grouped):
    listing_id = str(listing_id).strip()
    df_grouped['listing_id'] = df_grouped['listing_id'].astype(str).str.strip()

    try:
        return df_grouped[df_grouped['listing_id'] == listing_id].index[0]
    except IndexError:
        return None


# Helper function to convert a sentence into an embedding by averaging word embeddings
def sentence_to_embedding(sentence, model, embedding_dim=300):
    words = re.findall(r'\w+', sentence.lower())
    embedding_vectors = [model[word] for word in words if word in model]

    if len(embedding_vectors) == 0:
        return np.zeros(embedding_dim)

    return np.mean(embedding_vectors, axis=0)


# Stage 1: Get top recommendations based on textual similarity (description, comments, and name)
def get_text_based_recommendations(listing_id, df_grouped, textual_embeddings, num_recommendations=10):
    """
    Get top recommendations purely based on textual similarity (description, comments, and name).
    """
    listing_index = get_listing_index(listing_id, df_grouped)

    if listing_index is None:
        return pd.DataFrame()

    # Compute cosine similarity based on textual embeddings
    sim_scores = list(enumerate(cosine_similarity([textual_embeddings[listing_index]], textual_embeddings)[0]))

    # Sort similarity scores in descending order and select top recommendations
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get indices of similar listings, excluding the original listing itself
    similar_indices = [i[0] for i in sim_scores[1:num_recommendations + 1]]

    return df_grouped.iloc[similar_indices]


# Stage 2: Refine recommendations using structured features
def refine_with_structured_features(text_based_recommendations, structured_features, original_index,
                                    num_final_recommendations=5):
    """
    Refine top text-based recommendations based on structured features (e.g., bedrooms, price).
    """
    text_based_indices = text_based_recommendations.index
    text_based_structured_features = structured_features[text_based_indices]

    # Compute similarity based on structured features
    sim_scores_structured = cosine_similarity([structured_features[original_index]],
                                              text_based_structured_features).flatten()

    # Sort and get top N recommendations based on structured similarity
    top_structured_indices = np.argsort(sim_scores_structured)[-num_final_recommendations:]

    return text_based_recommendations.iloc[top_structured_indices]


# Function to build combined features for each listing using word embeddings and structured features
def build_combined_features(df):
    df_grouped = df.groupby('listing_id').agg({
        'comments': lambda x: ' '.join(x),
        'name': 'first',
        'description': 'first',
        'property_type': 'first',
        'bathrooms': 'first',
        'bedrooms': 'mean',
        'beds': 'mean',
        'minimum_nights': 'mean',
        'distance_to_center': 'mean',
        'amenities': lambda x: ','.join(set(','.join(x).split(','))),
        'number_of_reviews': 'mean',
    }).reset_index()

    # One-hot encode the 'property_type' column
    encoder = OneHotEncoder()
    property_type_encoded = encoder.fit_transform(df_grouped[['property_type']])

    # Combine textual data and convert it into embedding vectors (description, name, comments)
    df_grouped['textual_features'] = df_grouped['name'] + " " + df_grouped['description'] + " " + df_grouped['comments']
    textual_embeddings = np.array(
        [sentence_to_embedding(text, word2vec_model) for text in df_grouped['textual_features']])

    # Scale up the impact of textual embeddings by multiplying with the weight
    weighted_textual_embeddings = 5 * textual_embeddings

    # Structured features to be scaled
    structured_features = np.array(
        df_grouped[['bedrooms', 'beds', 'minimum_nights', 'distance_to_center', 'bathrooms']])
    scaler = StandardScaler()
    structured_features_scaled = scaler.fit_transform(structured_features)

    # Return textual and structured features separately
    return weighted_textual_embeddings, structured_features_scaled, df_grouped


# Two-stage recommendation system
def get_two_stage_recommendations(listing_id, df_grouped, textual_embeddings, structured_features,
                                  num_recommendations_text=10, num_final_recommendations=5):
    """
    Two-stage recommendation system: first text-based, then refined by structured features.
    """
    # Stage 1: Get top text-based recommendations
    text_based_recommendations = get_text_based_recommendations(listing_id, df_grouped, textual_embeddings,
                                                                num_recommendations=num_recommendations_text)

    # Stage 2: Refine top text-based recommendations using structured features
    original_index = get_listing_index(listing_id, df_grouped)
    final_recommendations = refine_with_structured_features(text_based_recommendations, structured_features,
                                                            original_index,
                                                            num_final_recommendations=num_final_recommendations)

    return final_recommendations
