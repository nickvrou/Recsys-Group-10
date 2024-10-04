import streamlit as st
import pandas as pd
import numpy as np
from content_based import build_combined_features, get_content_based_recommendations
from data_loader import load_data

def filter_listings_by_constraints(df, max_price, min_price, min_accommodates):
    # Filter based on price and accommodates
    filtered_df = df[(df['price'] >= min_price) & 
                     (df['price'] <= max_price) & 
                     (df['accommodates'] >= min_accommodates)]
    return filtered_df

df = load_data('C:/Users/vroun/Documents/GitHub/Recsys-Group-10/recommender/final_preprocessed_df (3).csv')

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
    # Convert user input into a dictionary
    user_input = {
        'listing_id': 'user_input',
        'review_scores_rating': 0,  # Not available from user
        'price': (price_range[0] + price_range[1]) / 2,  # Median price
        'neighbourhood_cleansed': ' ',  # Based on user location
        'property_type': ' ',  # Not asked, leave as None or default
        'amenities': ', '.join(amenities),  # Convert amenities list to string
        'comments': description,  # User description as comments
        'description': description,
        'name': ' ',  # No name, leave as None
        'bathrooms': bathrooms,  # Not asked
        'bedrooms': bedrooms,
        'beds': guests,  # Number of guests = number of beds
        'minimum_nights': (check_out - check_in).days,
        'maximum_nights': 0,  # Not available
        'distance_to_center': (distance_to_center[0] + distance_to_center[1]) / 2,
        'number_of_reviews': 0,
        'polarity': 0,  # Not available from user input
        'synthetic_rating': 0,  # Not available from user input
        'date': '1970-01-01'  # Not relevant for user input
    }

    # Append the user input as a new row to df_grouped for similarity comparison
    user_df = pd.DataFrame([user_input])

    df_with_user = pd.concat([df, user_df], ignore_index=False)

    df_with_user_c = filter_listings_by_constraints(df_with_user, price_range[1], price_range[0], guests)

    print(df_with_user_c.head())
    # Rebuild the combined features with the user input2
    combined_features = build_combined_features(df_with_user_c)

    # Use the user input as the base for recommendation
    listing_id_str = 'user_input'

    recommendations = get_content_based_recommendations(listing_id_str, df_with_user_c, combined_features)

    st.write("### Top Content-Based Recommendations:")
    if not recommendations.empty:
        st.write(recommendations[['name','description','review_scores_rating','bathrooms','bedrooms','distance_to_center','amenities', 'property_type', 'price']])
    else:
        st.write("No recommendations found for the given listing.")