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

# Load the preprocessed data and the original dataset

df_preprocessed = load_data('C:/Users/astratou/Downloads/final_preprocessed_df (3).csv')

df_original = pd.read_csv('C:/Users/astratou\Downloads\original_dataset.csv')

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
        'name': 'Custom Listing',  # Custom name for user input
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
    df_with_user = pd.concat([df_preprocessed, pd.DataFrame([user_input])], ignore_index=True)

    # Filter listings by constraints like price and number of guests
    df_with_user_c = filter_listings_by_constraints(df_with_user, price_range[1], price_range[0], guests)

    if df_with_user_c.empty:
        st.write("No listings found based on your constraints.")
    else:
        combined_features, df_grouped = build_combined_features(df_with_user_c)

        listing_id_str = 'user_input'

        recommendations = get_content_based_recommendations(listing_id_str, df_grouped, combined_features)

        if not recommendations.empty:
            # Get the recommended listing IDs
            recommended_listing_ids = recommendations['listing_id'].values

            # Find the original data for these listing IDs
            original_recommendations = df_original[df_original['listing_id'].isin(recommended_listing_ids)]

            # Truncate long text fields for better readability
            original_recommendations['description'] = original_recommendations['description'].apply(lambda x: x[:150] + '...' if len(x) > 150 else x)
            original_recommendations['amenities'] = original_recommendations['amenities'].apply(lambda x: x[:150] + '...' if len(x) > 150 else x)

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

            # Style the table using Pandas Styler
            def highlight_max(s):
                is_max = s == s.max()
                return ['background-color: lightgreen' if v else '' for v in is_max]

            styled_df = original_recommendations[['Name', 'Description', 'Review Score', 'Bathrooms',
                                                 'Bedrooms', 'Distance to Center (km)', 'Amenities',
                                                 'Property Type', 'Price (€)', 'Accommodates']].style.apply(highlight_max, subset=['Review Score'])

            # Set table styles for bold headers and remove index
            styled_df = styled_df.hide(axis="index").set_table_styles(
                [{'selector': 'th', 'props': [('font-weight', 'bold')]}]
            )

            # Display the table with Streamlit
            st.table(styled_df)
        else:
            st.write("No recommendations found for the given listing.")
