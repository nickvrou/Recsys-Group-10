import streamlit as st
import pandas as pd
import numpy as np
from content_based import build_combined_features, get_content_based_recommendations

# Sample Airbnb data (replace with your actual data)
data = {
    'listing_id': ['101', '102', '103', '104', '105'],
    'review_scores_rating': [4.5, 4.2, 3.8, 4.7, 4.1],
    'price': [150, 200, 100, 180, 130],
    'neighbourhood_cleansed': ['Centrum', 'West', 'East', 'South', 'North'],
    'property_type': ['Apartment', 'House', 'Apartment', 'House', 'Apartment'],
    'amenities': ['Wi-Fi, Kitchen', 'Wi-Fi, Parking', 'Kitchen, Parking', 'Wi-Fi', 'Wi-Fi, Kitchen, Parking'],
    'comments': ['Great place!', 'Nice stay', 'Good for family', 'Perfect location', 'Spacious and cozy'],
    'description': ['Cozy apartment', 'Spacious house', 'Family-friendly', 'Central location', 'Quiet area'],
    'name': ['Lovely Apt', 'Big House', 'Family Apt', 'Central Studio', 'Quiet Home']
}

# Create the DataFrame
df_grouped = pd.DataFrame(data)

# Add missing columns with placeholder values
missing_columns = [
    'availability_365', 'bathrooms', 'bedrooms', 'beds', 'distance_to_center', 
    'host_experience', 'host_is_superhost', 'host_total_listings_count', 
    'maximum_nights', 'minimum_nights', 'number_of_reviews', 'polarity', 
    'synthetic_rating', 'value_for_money'
]

# Add default values for each missing column
for column in missing_columns:
    if column not in df_grouped.columns:
        df_grouped[column] = 0  # Default for numeric columns

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
    listing_id = {
        'comments': None,
        'name': None,
        'description': description,  # User input description
        'property_type': None,
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
    st.write(listing_id)

    combined_features, df_grouped = build_combined_features(df_grouped)

    listing_id_str = '101'  
    
    recommendations = get_content_based_recommendations(listing_id_str, df_grouped, combined_features)

    st.write("### Top Content-Based Recommendations:")
    if not recommendations.empty:
        st.write(recommendations[['listing_id', 'neighbourhood_cleansed', 'price']])
    else:
        st.write("No recommendations found for the given listing.")
