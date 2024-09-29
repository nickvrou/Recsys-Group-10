import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds


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
    """
    Run SVD on the utility matrix and return predicted polarity values
    :param df: dataframe with known reviewer/listing pairs
    :param k: number of latent factors to keep for SVD
    :param user_col: user column name
    :param item_col: item column name
    :param value_col: value column name (e.g., polarity)
    :return: predicted matrix, users_index, and items_index
    """
    util_mat = create_utility_matrix(df, user_col, item_col, value_col)
    users_index, items_index = get_index_mappings(util_mat)

    # Mask NaN and fill them with item means
    mask = np.isnan(util_mat)
    masked_arr = np.ma.masked_array(util_mat, mask)
    item_means = np.mean(masked_arr, axis=0)
    util_mat = masked_arr.filled(item_means)

    # Demean utility matrix
    means = np.tile(item_means, (util_mat.shape[0], 1))
    util_mat_demeaned = util_mat - means

    # Perform SVD
    U, sigma, Vt = svds(util_mat_demeaned, k=k)
    sigma = np.diag(sigma)
    all_predicted_polarity = np.dot(np.dot(U, sigma), Vt) + means

    return all_predicted_polarity, users_index, items_index

def get_svd_recommendations(predictions, reviewer_id, users_index, items_index, listing_id_array,
                            num_recommendations=5):
    """
    Get top N recommendations for a reviewer based on SVD predictions
    :param predictions: predicted matrix from SVD
    :param reviewer_id: selected reviewer ID
    :param users_index: mapping from reviewer IDs to row indices
    :param items_index: mapping from item IDs to column indices
    :param listing_id_array: array of all listing IDs
    :param num_recommendations: number of top recommendations to return
    :return: DataFrame with top N recommendations
    """
    u_index = users_index[reviewer_id]
    item_indices = [items_index[listing_id_array[i]] for i in range(len(listing_id_array))]

    pred_user = [predictions[u_index, i_index] for i_index in item_indices]

    recommendations = pd.DataFrame({
        'listing_id': listing_id_array,
        'predicted_polarity': pred_user
    }).sort_values(by='predicted_polarity', ascending=False).head(num_recommendations)

    return recommendations

