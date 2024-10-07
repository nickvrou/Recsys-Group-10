import pandas as pd
import re

def load_data(file_path):
    """
    Loads and cleans the Airbnb listings data.
    - Removes missing values
    - Extracts numeric bathroom counts
    - Calculates a synthetic rating based on polarity
    """

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Remove rows with missing values
    df = df.dropna()

    # Extract numeric bathroom count from the text
    def extract_numeric_bathrooms(text):
        match = re.search(r'(\d+(\.\d+)?)', text)
        return float(match.group(1)) if match else 0.0

    # Apply the extraction function to 'bathrooms_text'
    df['bathrooms'] = df['bathrooms_text'].apply(extract_numeric_bathrooms)

    # Remove the 'bathrooms_text' column
    df = df.drop('bathrooms_text', axis=1)

    # Clean up listing ID by stripping extra spaces
    df['listing_id'] = df['listing_id'].astype(str).str.strip()

    # Calculate synthetic rating based on polarity (scaling from 1 to 5)
    df['synthetic_rating'] = df['polarity'].apply(lambda x: ((x + 1) / 2) * 4 + 1)

    return df
