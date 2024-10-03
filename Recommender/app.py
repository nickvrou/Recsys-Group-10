import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

from content_based import build_combined_features

# Utility matrix creation for SVD
def create_utility_matrix(df, user_col, item_col, value_col):
    """Create a pivot table utility matrix from the given dataframe"""
    return df.pivot_table(index=user_col, columns=item_col, values=value_col)

# Keep track of user and listing indices
def get_index_mappings(matrix):
    """Get mappings from user/listing IDs to row/column indices"""
    user_rows = list(matrix.index)
    listing_cols = list(matrix.columns)
    users_index = {user_rows[i]: i for i in range(len(user_rows))}
    items_index = {listing_cols[i]: i for i in range(len(listing_cols))}
    return users_index, items_index

# SVD for recommendation prediction
def recommend_predictions(df, k, user_col, item_col, value_col):
    """Run SVD on the utility matrix and return predicted polarity values"""
    util_mat = create_utility_matrix(df, user_col, item_col, value_col)
    users_index, items_index = get_index_mappings(util_mat)
    
    # Mask NaN and fill them with item means
    mask = np.isnan(util_mat)
    masked_arr = np.ma.masked_array(util_mat, mask)
    item_means = np.mean(masked_arr, axis=0)
    util_mat = masked_arr.filled(item_means)
    
    # Demean utility matrix
    util_mat_demeaned = util_mat - item_means
    
    # Perform SVD
    U, sigma, Vt = svds(util_mat_demeaned, k=k)
    sigma = np.diag(sigma)
    
    # Predicted matrix
    predicted_matrix = np.dot(np.dot(U, sigma), Vt) + item_means
    return predicted_matrix, users_index, items_index

# Sample DataFrame (replace with actual Airbnb data)
data = {
    'reviewer_id': [1, 2, 3, 1, 2, 3, 4],
    'listing_id': [101, 102, 103, 103, 101, 102, 101],
    'polarity': [5, 3, 4, 4, 5, 2, 3]
}
df_grouped = pd.DataFrame(data)

# Streamlit app layout
st.title("Airbnb Recommender System for a Family Getaway")

# Pre-fill the location and number of guests
location = st.text_input("Enter Location", "Amsterdam")
guests = st.number_input("Number of Guests", min_value=1, max_value=10, value=4)

# Select weekend dates (Check-in and Check-out)
check_in = st.date_input("Check-in Date", pd.to_datetime("today"))
check_out = st.date_input("Check-out Date", pd.to_datetime("today"))

# Number of bedrooms
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=5, value=2)

# Price range
price_range = st.slider("Price Range (per night)", 50, 500, (100, 300))

# Distance to center
distance_to_center = st.slider("Distance to center in km", 1, 20, (5,10))

# Description
description = st.text_input("Enter description")

# Amenities
amenities = st.multiselect(
    "Select Amenities",
    ["Wi-Fi", "Kitchen", "Parking", "Pet-friendly", "Air conditioning"],
    default=["Wi-Fi", "Kitchen"]
)

# Button to get recommendations
if st.button("Get Recommendations"):
    k = 2  # Latent factors for SVD (adjustable)
    
    
    # Using content-based recommendation logic from content_based.py

    # Features include neighbourhood, property type, and amenities
    def content_based_recommendations(dataframe, user_preferences):
        # Use the build_combined_features function from the content_based.py
        combined_features = build_combined_features(dataframe)
        
        # Calculate cosine similarity between user preferences and listings
        similarity_matrix = cosine_similarity(user_preferences, combined_features)
        
        # Sort and return top N recommendations
        top_recommendations_idx = np.argsort(similarity_matrix[0])[::-1][:3]
        return dataframe.iloc[top_recommendations_idx]

    # Build combined features for listings
    user_preferences = [[1, 0, 0, 1]]  # Simulating user preference input
    recommendations = content_based_recommendations(df_grouped, user_preferences)

    # Display top recommendations to the user
    st.write("### Top Content-Based Recommendations:")
    st.write(recommendations[['listing_id', 'neighbourhood_cleansed', 'price']])
    
    predicted_matrix, users_index, items_index = recommend_predictions(df_grouped, k, 'reviewer_id', 'listing_id', 'polarity')
    
    # Simulate showing top recommendations
    st.write("### Recommendations:")
    user_id = 1 
    
    if user_id in users_index:
        user_idx = users_index[user_id]
        user_predictions = predicted_matrix[user_idx, :]
        
        # Get top N recommendations
        top_recommendations = np.argsort(user_predictions)[::-1][:3]
        for idx in top_recommendations:
            listing_id = list(items_index.keys())[list(items_index.values()).index(idx)]
            st.write(f"Recommended Airbnb listing ID: {listing_id}")
    else:
        st.write("No recommendations available for this user.")
