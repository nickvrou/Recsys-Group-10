from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def get_listing_index(listing_id, df_grouped):
    try:
        return df_grouped[df_grouped['listing_id'] == listing_id].index[0]
    except IndexError:
        return None

def build_combined_features(df):

    df_grouped = df.groupby('listing_id').agg({
        'comments': lambda x: ' '.join(x),
        'name': 'first',
        'description': 'first',
        'neighbourhood_cleansed': 'first',
        'property_type': 'first',
        'review_scores_rating': 'mean',
        'bathrooms': 'first',
        'bedrooms': 'mean',
        'beds': 'mean',
        'minimum_nights': 'mean',
        'maximum_nights': 'mean',
        'distance_to_center': 'mean',
        'polarity': 'mean',
        'synthetic_rating': 'first',
        'amenities': lambda x: ','.join(set(','.join(x).split(','))),
        'number_of_reviews': 'mean',
        'price': 'mean'
    }).reset_index()

    df_grouped['listing_id'] = df_grouped['listing_id'].astype(str).str.strip()

    encoder = OneHotEncoder()

    neighbourhood_encoded = encoder.fit_transform(df_grouped[['neighbourhood_cleansed']])

    property_type_encoded = encoder.fit_transform(df_grouped[['property_type']])

    # Processing amenities with TF-IDF
    df_grouped['amenities'] = df_grouped['amenities'].apply(lambda x: ','.join(x.split(',')))
    vectorizer_amenities = TfidfVectorizer(tokenizer=lambda x: x.split(','), lowercase=False)
    amenities_matrix = vectorizer_amenities.fit_transform(df_grouped['amenities'])

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))

    tfidf_matrix = vectorizer.fit_transform(
        df_grouped['name'] + " " + df_grouped['description'] + " " + df_grouped['comments'])

    # Scaling structured features
    structured_features = np.array(df_grouped[['review_scores_rating', 'bedrooms', 'beds',
                                               'minimum_nights', 'maximum_nights', 'distance_to_center',
                                               'polarity', 'bathrooms', 'number_of_reviews',
                                               'price']])
    scaler = StandardScaler()

    structured_features_scaled = scaler.fit_transform(structured_features)

    # Combine all features into a sparse matrix
    combined_features = hstack(
        [tfidf_matrix, neighbourhood_encoded, property_type_encoded, amenities_matrix, structured_features_scaled])

    # Convert to CSR format for slicing
    return combined_features.tocsr()


# Content-Based Recommendation Engine
def get_content_based_recommendations(listing_id, df_grouped, combined_features, num_recommendations=5):

    listing_index = get_listing_index(listing_id, df_grouped)
    print(listing_index)

    if listing_index is None:

        return pd.DataFrame()


    # Now combined_features is in CSR format, and slicing will work
    sim_scores = list(enumerate(cosine_similarity(combined_features[listing_index], combined_features)[0]))

    # Sort similarity scores in descending order and pick top recommendations
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    similar_indices = [i[0] for i in sim_scores[1:num_recommendations + 1]]

    recommendations = df_grouped.iloc[similar_indices]

    return recommendations
