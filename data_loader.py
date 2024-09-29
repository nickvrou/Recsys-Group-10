import pandas as pd

def load_data(file_path):
    """
    Load the preprocessed data from a CSV or other format.
    Assumes that the data is already preprocessed.

    :param file_path: Path to the data file
    :return: pandas DataFrame
    """
    return pd.read_csv(file_path)
