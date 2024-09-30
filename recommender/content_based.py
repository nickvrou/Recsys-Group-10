from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from recommender.data_loader import load_data

def get_listing_index(listing_id, df_grouped):
    try:
        return df_grouped[df_grouped['listing_id'] == listing_id].index[0]
    except IndexError:
        return None

def build_combined_features(df_grouped):

    df_grouped['listing_id'] = df_grouped['listing_id'].astype(str).str.strip()

    df_grouped['value_for_money'] = df_grouped['review_scores_rating'] / df_grouped['price']

    df_grouped['host_is_superhost'] = df_grouped['host_is_superhost'].apply(lambda x: 1 if x == 't' else 0)

    encoder = OneHotEncoder()
    neighbourhood_encoded = encoder.fit_transform(df_grouped[['neighbourhood_cleansed']])
    property_type_encoded = encoder.fit_transform(df_grouped[['property_type']])

    # Processing amenities with TF-IDF
    df_grouped['amenities'] = df_grouped['amenities'].apply(lambda x: ','.join(x.split(',')))
    vectorizer_amenities = TfidfVectorizer(tokenizer=lambda x: x.split(','), lowercase=False)
    amenities_matrix = vectorizer_amenities.fit_transform(df_grouped['amenities'])

    # Scaling structured features
    structured_features = np.array(df_grouped[['review_scores_rating', 'bedrooms', 'beds',
                                               'minimum_nights', 'maximum_nights', 'distance_to_center',
                                               'polarity', 'bathrooms', 'host_experience',
                                               'host_is_superhost', 'host_total_listings_count', 'number_of_reviews',
                                               'price', 'availability_365', 'value_for_money']])
    scaler = StandardScaler()
    structured_features_scaled = scaler.fit_transform(structured_features)

    # Combine all features into a sparse matrix
    combined_features = hstack(
        [neighbourhood_encoded, property_type_encoded, amenities_matrix, structured_features_scaled])

    return combined_features.tocsr()


# Content-Based Recommendation Engine
def get_content_based_recommendations(listing_id, df_grouped, combined_features, num_recommendations=5,
                                      preferences=None):
    listing_index = get_listing_index(listing_id, df_grouped)

    if listing_index is None:
        return pd.DataFrame()

    # Calculate cosine similarity for the target listing
    sim_scores = list(enumerate(cosine_similarity(combined_features[listing_index], combined_features)[0]))

    # Sort similarity scores in descending order and pick top recommendations
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    similar_indices = [i[0] for i in sim_scores[1:num_recommendations + 1]]

    recommendations = df_grouped.iloc[similar_indices]

    return recommendations

