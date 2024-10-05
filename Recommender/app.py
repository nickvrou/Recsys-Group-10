import streamlit as st
import pandas as pd
from content_based import build_combined_features, get_content_based_recommendations
from data_loader import load_data

def filter_listings_by_constraints(df, max_price, min_price, min_accommodates):
    # Filter based on price and accommodates
    filtered_df = df[(df['price'] >= min_price) & 
                     (df['price'] <= max_price) & 
                     (df['accommodates'] >= min_accommodates)]
    return filtered_df

# Load the data
df = load_data('C:/Users/astratou/Downloads/final_preprocessed_df (3).csv')

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
        'review_scores_rating': 5,  # Not available from user
        'price': (price_range[0] + price_range[1]) / 2,  # Median price
        'neighbourhood_cleansed': 'Center',  # Based on user location
        'property_type': 'Apartment',  # Not asked, leave as None or default
        'amenities': ', '.join(amenities),  # Convert amenities list to string
        'comments': description,  # User description as comments
        'description': description,
        'name': 'Ma',  # No name, leave as None
        'bathrooms': bathrooms,
        'bedrooms': bedrooms,
        'beds': guests,  # Number of guests = number of beds
        'minimum_nights': (check_out - check_in).days,
        'maximum_nights': 5,  # Not available
        'distance_to_center': (distance_to_center[0] + distance_to_center[1]) / 2,
        'number_of_reviews': 100,
        'polarity': 0.9,
        'synthetic_rating': 5,
        'date': '2024-01-01',
        'accommodates': guests
    }

    # Append the user input as a new row to df_grouped for similarity comparison
    df_with_user = pd.concat([df, pd.DataFrame([user_input])], ignore_index=True)

    # Filter listings by constraints like price and number of guests
    df_with_user_c = filter_listings_by_constraints(df_with_user, price_range[1], price_range[0], guests)

    if df_with_user_c.empty:

        st.write("No listings found based on your constraints.")

    else:

        combined_features, df_grouped = build_combined_features(df_with_user_c)

        listing_id_str = 'user_input'

        recommendations = get_content_based_recommendations(listing_id_str, df_grouped, combined_features)

        st.write("### Top Content-Based Recommendations:")

        if not recommendations.empty:
            st.write(recommendations[['name', 'description', 'review_scores_rating', 'bathrooms', 'bedrooms', 'distance_to_center', 'amenities', 'property_type', 'price', 'accommodates']])
        else:
            st.write("No recommendations found for the given listing.")
