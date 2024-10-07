import streamlit as st
import pandas as pd
from explainations import filter_listings_by_constraints, remove_stop_words, explain_recommendation_with_word2vec
from content_based import build_combined_features, get_two_stage_recommendations
from data_loader import load_data
from gensim.models import KeyedVectors
from popularity_filtering import calculate_popularity_score

# Load the pre-trained Word2Vec model
word2vec_model = KeyedVectors.load_word2vec_format(
    "C:/Users/astratou/Downloads/GoogleNews-vectors-negative300.bin.gz", binary=True)

# Define a list of stop words to be removed from user input
stop_words = [
    'the', 'is', 'in', 'and', 'to', 'a', 'of', 'it', 'we', 'was', 'for', 'it',
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

# Load preprocessed and original datasets
df_preprocessed = load_data('C:/Users/astratou/Downloads/final_preprocessed_df (3).csv')
df_original = pd.read_csv('C:/Users/astratou/Downloads/original_dataset.csv')
df_original['listing_id'] = df_original['listing_id'].astype(str).str.strip()

# Streamlit UI layout
st.title("Airbnb Recommender System for a Family Getaway")

# Inputs for the user to specify details
location = st.text_input("Enter Location", "Amsterdam")
guests = st.number_input("Number of Guests", min_value=1, max_value=10, value=4)
check_in = st.date_input("Check-in Date", pd.to_datetime("today"))
check_out = st.date_input("Check-out Date", pd.to_datetime("today"))
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=5, value=2)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10)
price_range = st.slider("Price Range (per night)", 50, 500, (100, 300))
distance_to_center = st.slider("Distance to center in km", 1, 20, (5, 10))
description = st.text_input("Enter description")
amenities = st.multiselect(
    "Select Amenities",
    ["Wi-Fi", "Kitchen", "Parking", "Pet-friendly", "Air conditioning"],
    default=["Wi-Fi", "Kitchen"]
)

# Filter stop words from the user's input
filtered_description = remove_stop_words(description, stop_words)

if st.button("Get Recommendations"):
    # Create user input as a new listing
    user_input = {
        'listing_id': 'user_input',
        'price': (price_range[0] + price_range[1]) / 2,
        'neighbourhood_cleansed': 'Center',
        'property_type': 'Apartment',
        'amenities': ', '.join(amenities),
        'comments': description,
        'description': description,
        'name': 'Custom Listing',
        'bathrooms': bathrooms,
        'bedrooms': bedrooms,
        'beds': guests,
        'minimum_nights': (check_out - check_in).days,
        'distance_to_center': (distance_to_center[0] + distance_to_center[1]) / 2,
        'accommodates': guests
    }

    # Append user input to the preprocessed dataset
    df_with_user = pd.concat([df_preprocessed, pd.DataFrame([user_input])], ignore_index=True)

    # Apply constraints such as price and the number of guests
    df_with_user_c = filter_listings_by_constraints(df_with_user, price_range[1], price_range[0], guests)

    if df_with_user_c.empty:
        st.write("No listings found based on your constraints.")
    else:
        # Get features for the recommendation engine
        textual_embeddings, structured_features, df_grouped = build_combined_features(df_with_user_c)
        listing_id_str = 'user_input'

        # Use the two-stage recommendation system (textual + structured data)
        recommendations = get_two_stage_recommendations(
            listing_id_str, df_grouped, textual_embeddings, structured_features)

        if not recommendations.empty:
            # Fetch the original listings based on recommendation IDs
            recommended_listing_ids = recommendations['listing_id'].values
            original_recommendations = df_original[df_original['listing_id'].isin(recommended_listing_ids)]

            # Clean data by converting some fields to integers
            original_recommendations['bathrooms'] = original_recommendations['bathrooms'].astype(int)
            original_recommendations['bedrooms'] = original_recommendations['bedrooms'].astype(int)
            original_recommendations['distance_to_center'] = original_recommendations['distance_to_center'].astype(int)
            original_recommendations['price'] = original_recommendations['price'].astype(int)

            # Rename columns to more professional and readable names
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

            # Merge polarity score and sort by it
            original_recommendations = original_recommendations.merge(df_preprocessed[['listing_id', 'polarity']],
                                                                      on='listing_id',
                                                                      how='left')
            original_recommendations['polarity'] = original_recommendations['polarity'].fillna(0)
            original_recommendations = original_recommendations.sort_values(by='polarity', ascending=False)
            original_recommendations.drop_duplicates(subset=['listing_id'], inplace=True)

            # Generate explanations for each recommendation
            for _, rec in original_recommendations.iterrows():
                filtered_description_od = remove_stop_words(rec['Description'], stop_words)
                filtered_comments_od = remove_stop_words(rec['comments'], stop_words)
                polarity_score = rec['polarity']
                explanation = explain_recommendation_with_word2vec(
                    filtered_description, filtered_description_od, filtered_comments_od,
                    word2vec_model, polarity_score
                )
                st.write(f"We recommend '{rec['Name']}' because {explanation}")

            # Display the recommendations in a table
            st.dataframe(original_recommendations[['Name', 'Description', 'Review Score', 'Bathrooms',
                                                   'Bedrooms', 'Distance to Center (km)',
                                                   'Property Type', 'Price (€)', 'Accommodates']])
        else:
            st.write("No recommendations found for the given listing.")
