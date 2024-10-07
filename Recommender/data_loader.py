import pandas as pd
import re

def load_data(file_path):
    """
    Load the preprocessed data from a CSV or other format.
    Assumes that the data is already preprocessed.

    :param file_path: Path to the data file
    :return: pandas DataFrame
    """

    #Read csv
    df = pd.read_csv(file_path)

    #Drop
    df = df.dropna()

    def extract_numeric_bathrooms(text):

        match = re.search(r'(\d+(\.\d+)?)', text)

        if match:

            return float(match.group(1))

        return 0.0

    df['bathrooms'] = df['bathrooms_text'].apply(extract_numeric_bathrooms)

    df = df.drop('bathrooms_text', axis=1)

    df['listing_id'] = df['listing_id'].astype(str).str.strip()

    df['synthetic_rating'] = df['polarity'].apply(lambda x: ((x + 1) / 2) * 4 + 1)

    return df
