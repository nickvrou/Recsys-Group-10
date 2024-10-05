import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Helper function to get the index of a listing based on its ID
def get_listing_index(listing_id, df_grouped):
    try:
        return df_grouped[df_grouped['listing_id'] == listing_id].index[0]
    except IndexError:
        return None

# Function to build combined features for each listing
def build_combined_features(df):
    df_grouped = df.groupby('listing_id').agg({
        'comments': lambda x: ' '.join(x),
        'name': 'first',
        'description': 'first',
        'property_type': 'first',
        'review_scores_rating': 'mean',
        'bathrooms': 'first',
        'bedrooms': 'mean',
        'beds': 'mean',
        'minimum_nights': 'mean',
        'maximum_nights': 'mean',
        'distance_to_center': 'mean',
        'polarity': 'mean',
        'synthetic_rating': 'mean',
        'amenities': lambda x: ','.join(set(','.join(x).split(','))),
        'number_of_reviews': 'mean',
        'date': 'first'
    }).reset_index()

    df_grouped['listing_id'] = df_grouped['listing_id'].astype(str).str.strip()
    reference_date = pd.to_datetime('1970-01-01')
    df_grouped['date'] = (pd.to_datetime(df_grouped['date']) - reference_date).dt.days

    encoder = OneHotEncoder()
    property_type_encoded = encoder.fit_transform(df_grouped[['property_type']])

    # Processing amenities with TF-IDF
    df_grouped['amenities'] = df_grouped['amenities'].apply(lambda x: ','.join(x.split(',')))
    vectorizer_amenities = TfidfVectorizer(tokenizer=lambda x: x.split(','), lowercase=False, token_pattern=None)
    amenities_matrix = vectorizer_amenities.fit_transform(df_grouped['amenities'])

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df_grouped['name'] + " " + df_grouped['description'] + " " + df_grouped['comments'])

    # Scaling structured features
    structured_features = np.array(df_grouped[['review_scores_rating', 'bedrooms', 'beds', 'minimum_nights', 
                                               'maximum_nights', 'distance_to_center', 'polarity', 'bathrooms', 'date']])
    scaler = StandardScaler()
    structured_features_scaled = scaler.fit_transform(structured_features)

    # Combine all features into a sparse matrix
    combined_features = hstack([tfidf_matrix, property_type_encoded, amenities_matrix, structured_features_scaled])

    return combined_features.tocsr(), df_grouped


# Content-Based Recommendation Engine
def get_content_based_recommendations(listing_id, df_grouped, combined_features, num_recommendations=5):
    listing_index = get_listing_index(listing_id, df_grouped)

    if listing_index is None:
        return pd.DataFrame()

    # Now combined_features is in CSR format, and slicing will work
    sim_scores = list(enumerate(cosine_similarity(combined_features[listing_index], combined_features)[0]))

    # Sort similarity scores in descending order and pick top recommendations
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    similar_indices = [i[0] for i in sim_scores[1:num_recommendations + 1]]

    recommendations = df_grouped.iloc[similar_indices]

    return recommendations


# MMR (Maximal Marginal Relevance) for diverse and relevant recommendations
def mmr(query_embedding, item_embeddings, already_selected, lambda_param, num_recommendations=5):
    """
    Apply Maximal Marginal Relevance (MMR) to select diverse and relevant recommendations.
    """
    selected_items = []
    item_indices = list(range(item_embeddings.shape[0]))

    for _ in range(num_recommendations):
        remaining_items = list(set(item_indices) - set(already_selected) - set(selected_items))
        if len(remaining_items) == 0:
            break

        # Calculate relevance (similarity to the query)
        relevance_scores = cosine_similarity(query_embedding.reshape(1, -1), item_embeddings[remaining_items]).flatten()

        # Calculate diversity (similarity to already selected items)
        if len(selected_items) > 0:
            diversity_scores = np.max(cosine_similarity(item_embeddings[remaining_items], item_embeddings[selected_items]), axis=1)
        else:
            diversity_scores = np.zeros(len(remaining_items))  # No diversity score if nothing selected yet

        # Maximize MMR: lambda * relevance - (1 - lambda) * diversity
        mmr_scores = lambda_param * relevance_scores - (1 - lambda_param) * diversity_scores
        selected_item_idx = remaining_items[np.argmax(mmr_scores)]

        # Add the selected item to the recommendations
        selected_items.append(selected_item_idx)

    return selected_items

# Apply MMR to get diverse recommendations
def get_mmr_recommendations(listing_id, df_grouped, feature_matrix, lambda_param=0.8, num_recommendations=5):
    listing_index = get_listing_index(listing_id, df_grouped)

    if listing_index is None:
        return pd.DataFrame()

    # Get the embedding of the query item (the item for which recommendations are being made)
    query_embedding = feature_matrix[listing_index]

    # Apply MMR to get the most relevant and diverse items
    recommended_indices = mmr(query_embedding, feature_matrix, already_selected=[listing_index], lambda_param=lambda_param, num_recommendations=num_recommendations)

    recommendations = df_grouped.iloc[recommended_indices]
    return recommendations