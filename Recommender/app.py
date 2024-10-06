import streamlit as st
import pandas as pd
from explainations import filter_listings_by_constraints, remove_stop_words, explain_recommendation_with_word2vec
from content_based import build_combined_features, get_two_stage_recommendations
from data_loader import load_data
from gensim.models import KeyedVectors

# Load Word2Vec model
word2vec_model = KeyedVectors.load_word2vec_format("C:/Users/astratou/Downloads/GoogleNews-vectors-negative300.bin.gz", binary=True)

# List of stop words
stop_words = [
    'the', 'is', 'in', 'and', 'to', 'a', 'of', 'it', 'we', 'was', 'for', 'It',
    'but', 'on', 'with', 'as', 'you', 'at', 'this', 'that', 'had', 'our',
    'be', 'by', 'or', 'an', 'are', 'from', 'so', 'if', 'have', 'my',
    'they', 'which', 'one', 'their', 'there', 'what', 'more', 'when',
    'can', 'your', 'will', 'would', 'should', 'could', 'about', 'out', 'up',
    'them', 'some', 'me', 'just', 'into', 'has', 'also', 'very', 'been',
    'did', 'do', 'he', 'she', 'his', 'her', 'how', 'then', 'than', 'other',
    'over', 'because', 'any', 'only', 'were', 'after', 'did', 'these',
    'who', 'its', 'see', 'well', 'here', 'get', 'got', 'even', 'make',
    'made', 'us', 'you', 'your', 'yours', 'I', 'am', 'he', 'she', 'it',
    'we', 'they', 'you', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'myself', 'ourselves', 'yourself', 'yourselves',
    'himself', 'herself', 'itself', 'themselves', 'each', 'few', 'many',
    'some', 'such', 'very', 's', 't', 'can', 'will', 'don', 'should', 'now'
]

# Load the preprocessed data and the original dataset
df_preprocessed = load_data('C:/Users/astratou/Downloads/final_preprocessed_df (3).csv')
df_original = pd.read_csv('C:/Users/astratou/Downloads/original_dataset.csv')
df_original['listing_id'] = df_original['listing_id'].astype(str).str.strip()

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
# Number of bathrooms
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10)

# Price range
price_range = st.slider("Price Range (per night)", 50, 500, (100, 300))

# Distance to center
distance_to_center = st.slider("Distance to center in km", 1, 20, (5, 10))

# Description
description = st.text_input("Enter description")

# Remove stop words from the user's input
filtered_description = remove_stop_words(description, stop_words)

# Amenities
amenities = st.multiselect(
    "Select Amenities",
    ["Wi-Fi", "Kitchen", "Parking", "Pet-friendly", "Air conditioning"],
    default=["Wi-Fi", "Kitchen"]
)

if st.button("Get Recommendations"):

    # Create the user input as a new listing
    user_input = {
        'listing_id': 'user_input',
        'price': (price_range[0] + price_range[1]) / 2,  # Median price
        'neighbourhood_cleansed': 'Center',  # Based on user location
        'property_type': 'Apartment',  # Not asked, leave as None or default
        'amenities': ', '.join(amenities),  # Convert amenities list to string
        'comments': description,  # User description as comments
        'description': description,
        'name': 'Custom Listing',  # Custom name for user input
        'bathrooms': bathrooms,
        'bedrooms': bedrooms,
        'beds': guests,  # Number of guests = number of beds
        'minimum_nights': (check_out - check_in).days,
        'distance_to_center': (distance_to_center[0] + distance_to_center[1]) / 2,
        'accommodates': guests
    }

    # Append the user input as a new row to df_grouped for similarity comparison
    df_with_user = pd.concat([df_preprocessed, pd.DataFrame([user_input])], ignore_index=True)

    # Filter listings by constraints like price and number of guests
    df_with_user_c = filter_listings_by_constraints(df_with_user, price_range[1], price_range[0], guests)

    if df_with_user_c.empty:
        st.write("No listings found based on your constraints.")
    else:
        # Unpack textual and structured features separately
        textual_embeddings, structured_features, df_grouped = build_combined_features(df_with_user_c)

        listing_id_str = 'user_input'

        # Now pass both textual_embeddings and structured_features to get_two_stage_recommendations
        recommendations = get_two_stage_recommendations(listing_id_str, df_grouped, textual_embeddings, structured_features)

        if not recommendations.empty:
            # Get the recommended listing IDs
            recommended_listing_ids = recommendations['listing_id'].values

            # Find the original data for these listing IDs
            original_recommendations = df_original[df_original['listing_id'].isin(recommended_listing_ids)]

            # Truncate long text fields for better readability
            original_recommendations['description'] = original_recommendations['description'].apply(
                lambda x: x[:150] + '...' if len(x) > 150 else x)
            original_recommendations['amenities'] = original_recommendations['amenities'].apply(
                lambda x: x[:150] + '...' if len(x) > 150 else x)
            original_recommendations['comments'] = original_recommendations['comments'].apply(
                lambda x: x[:150] + '...' if len(x) > 150 else x)

            # Convert bathrooms, bedrooms, distance_to_center, and price to integers
            original_recommendations['bathrooms'] = original_recommendations['bathrooms'].astype(int)
            original_recommendations['bedrooms'] = original_recommendations['bedrooms'].astype(int)
            original_recommendations['distance_to_center'] = original_recommendations['distance_to_center'].astype(int)
            original_recommendations['price'] = original_recommendations['price'].astype(int)

            # Rename the columns with capitalized and professional names
            original_recommendations = original_recommendations.rename(columns={
                'name': 'Name',
                'description': 'Description',
                'review_scores_rating': 'Review Score',
                'bathrooms': 'Bathrooms',
                'bedrooms': 'Bedrooms',
                'distance_to_center': 'Distance to Center (km)',
                'amenities': 'Amenities',
                'property_type': 'Property Type',
                'price': 'Price (€)',
                'accommodates': 'Accommodates'
            })

            original_recommendations.drop_duplicates(subset=['listing_id'], inplace=True)

            # Generate explanations for recommendations using Word2Vec-based semantic similarity
            for _, rec in original_recommendations.iterrows():
                filtered_description_od = remove_stop_words(rec['Description'], stop_words)
                filtered_comments_od = remove_stop_words(rec['comments'], stop_words)
                popularity_score = rec.get('popularity_score', None)  # Replace with actual logic for popularity score
                explanation = explain_recommendation_with_word2vec(
                    filtered_description,  # The user's input (after removing stop words)
                    filtered_description_od,  # Listing's description
                    filtered_comments_od,  # Listing's comments
                    word2vec_model,  # Word2Vec model
                    popularity_score  # Include the popularity score
                )
                st.write(f"We recommend '{rec['Name']}' because {explanation}")

            # Style the table using Pandas Styler
            styled_df = st.dataframe(original_recommendations[['Name', 'Description', 'Review Score', 'Bathrooms',
                                                               'Bedrooms', 'Distance to Center (km)',
                                                               'Property Type', 'Price (€)', 'Accommodates']],use_container_width=True)

        else:
            st.write("No recommendations found for the given listing.")