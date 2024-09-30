import pandas as pd
import re

def load_data(file_path):
    """
    Load the preprocessed data from a CSV or other format.
    Assumes that the data is already preprocessed.

    :param file_path: Path to the data file
    :return: pandas DataFrame
    """

    df = pd.read_csv(file_path)

    df = df.dropna()

    def extract_numeric_bathrooms(text):
        match = re.search(r'(\d+(\.\d+)?)', text)
        if match:
            return float(match.group(1))
        return 0.0

    df['bathrooms'] = df['bathrooms_text'].apply(extract_numeric_bathrooms)

    df = df.drop('bathrooms_text', axis=1)

    df['host_since'] = pd.to_datetime(df['host_since'])

    df['host_experience'] = 2024 - df['host_since'].dt.year

    df['host_experience'] = df['host_experience'].fillna(0)

    df['value_for_money'] = df['review_scores_rating'] / df['price']

    df['host_is_superhost'] = df['host_is_superhost'].apply(lambda x: 1 if x == 't' else 0)

    return df
