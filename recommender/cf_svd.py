import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


# Utility matrix creation for SVD
def create_utility_matrix(df, user_col, item_col, value_col):
    """
    Create a pivot table utility matrix from the given dataframe
    :param df: Input dataframe
    :param user_col: Column representing user IDs (e.g., 'reviewer_id')
    :param item_col: Column representing item IDs (e.g., 'listing_id')
    :param value_col: Column representing the value (e.g., 'polarity')
    :return: utility matrix (user-item matrix)
    """
    return df.pivot_table(index=user_col, columns=item_col, values=value_col)


# Keep track of user and listing indices
def get_index_mappings(matrix):
    """
    Get mappings from user/listing IDs to row/column indices
    :param matrix: utility matrix
    :return: dictionaries mapping users to rows and listings to columns
    """
    user_rows = list(matrix.index)
    listing_cols = list(matrix.columns)
    users_index = {user_rows[i]: i for i in range(len(user_rows))}
    items_index = {listing_cols[i]: i for i in range(len(listing_cols))}
    return users_index, items_index


# SVD for recommendation prediction
def recommend_predictions(df, k, user_col, item_col, value_col):
    # Create the utility matrix
    util_mat = create_utility_matrix(df, user_col, item_col, value_col)

    # Handle missing values and fill them with item means
    mask = np.isnan(util_mat)
    masked_arr = np.ma.masked_array(util_mat, mask)
    item_means = np.mean(masked_arr, axis=0)
    util_mat = masked_arr.filled(item_means)

    # Demean the utility matrix
    means = np.tile(item_means, (util_mat.shape[0], 1))
    util_mat_demeaned = util_mat - means

    # Perform truncated SVD
    svd = TruncatedSVD(n_components=k, random_state=42)

    U = svd.fit_transform(util_mat_demeaned)  # User latent factors
    Vt = svd.components_  # Item latent factors

    users_index, items_index = get_index_mappings(util_mat)
    return U, Vt, users_index, items_index, item_means


# Get top N SVD-based recommendations
def get_svd_recommendations(U, Vt, user_latent_factors, items_index, item_means, listing_id_array,
                            num_recommendations=5):
    """
    Get top N recommendations for a reviewer based on SVD predictions
    :param U: User latent factors from SVD
    :param Vt: Item latent factors from SVD
    :param user_latent_factors: User latent factor vector for the reviewer
    :param items_index: mapping from item IDs to column indices
    :param item_means: The mean values of items for filling missing values
    :param listing_id_array: array of all listing IDs
    :param num_recommendations: number of top recommendations to return
    :return: DataFrame with top N recommendations
    """
    # Calculate predicted ratings for all items for the given user
    pred_user = np.dot(user_latent_factors, Vt) + item_means

    # Create a DataFrame with listing IDs and predicted polarity values
    recommendations = pd.DataFrame({
        'listing_id': listing_id_array,
        'predicted_polarity': pred_user
    }).sort_values(by='predicted_polarity', ascending=False).head(num_recommendations)

    return recommendations

