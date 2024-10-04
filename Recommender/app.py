import streamlit as st
import pandas as pd
import numpy as np
from content_based import build_combined_features, get_content_based_recommendations
from data_loader import load_data

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
   
     # Assign the actual values for the required fields, and set the rest to None
    user_listing = {
        'listing_id': '6969',
        'comments': '',
        'name': '',
        'description': description,  # User input description
        'property_type': '',
        'review_scores_rating': None,
        'bathrooms': None,
        'bedrooms': bedrooms,  # Number of bedrooms input by the user
        'beds': guests,  # Number of guests is mapped to beds
        'minimum_nights': (check_out - check_in).days,  # Days between check-in and check-out
        'maximum_nights': None,
        'distance_to_center': (distance_to_center[0] + distance_to_center[1]) / 2,  # Median of distance range
        'polarity': None,
        'synthetic_rating': None,
        'amenities': ', '.join(amenities),  # Selected amenities
        'number_of_reviews': None,
        'price': (price_range[0] + price_range[1]) / 2,  # Median of price range
        'date': None
    }

    st.write("### User Input for Listing:")
    st.write(user_listing)

    user_listing_df = pd.DataFrame([user_listing])

    df = pd.concat([user_listing_df, df], ignore_index=True)

    combined_features = build_combined_features(df)

    listing_id = '6969' 
    
    recommendations = get_content_based_recommendations(listing_id, df, combined_features)

    st.write("### Top Content-Based Recommendations:")
    if not recommendations.empty:
        st.write(recommendations[['listing_id', 'neighbourhood_cleansed','review_scores_rating', 'price']])
    else:
        st.write("No recommendations found for the given listing.")
