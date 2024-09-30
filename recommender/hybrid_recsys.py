import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from recommender.content_based import get_content_based_recommendations
from recommender.cf_svd import get_svd_recommendations


def get_hybrid_recommendations(listing_id, reviewer_id, df_grouped, combined_features, svd_predictions, users_index,
                               items_index, alpha=0.5, num_recommendations=5):
    """
    Hybrid recommender that combines content-based and collaborative filtering recommendations.

    :param listing_id: The listing_id to base content-based recommendations on
    :param reviewer_id: The reviewer_id for collaborative filtering recommendations
    :param df_grouped: The preprocessed DataFrame with listing data
    :param combined_features: Feature matrix used for content-based filtering
    :param svd_predictions: Predicted matrix from SVD
    :param users_index: Mapping from user IDs to row indices (SVD)
    :param items_index: Mapping from listing IDs to column indices (SVD)
    :param alpha: Weight for combining SVD and content-based scores (0 <= alpha <= 1)
    :param num_recommendations: Number of top recommendations to return
    :return: DataFrame with hybrid recommendations
    """

    # Get Content-Based Recommendations
    content_based_recs = get_content_based_recommendations(listing_id, df_grouped, combined_features,
                                                           num_recommendations)
    content_based_recs['cb_score'] = cosine_similarity(
        combined_features[get_listing_index(listing_id, df_grouped)],
        combined_features[content_based_recs.index]).flatten()

    # Get SVD-Based Recommendations
    svd_recs = get_svd_recommendations(svd_predictions, reviewer_id, users_index, items_index,
                                       df_grouped['listing_id'].unique(), num_recommendations)

    # Merge the recommendations based on listing_id
    hybrid_recs = pd.merge(content_based_recs[['listing_id', 'cb_score']],
                           svd_recs[['listing_id', 'predicted_polarity']], on='listing_id', how='inner')

    # Normalize the collaborative filtering score (predicted polarity)
    hybrid_recs['cf_score'] = hybrid_recs['predicted_polarity'] / hybrid_recs['predicted_polarity'].max()

    # Calculate the hybrid score (weighted combination)
    hybrid_recs['hybrid_score'] = alpha * hybrid_recs['cf_score'] + (1 - alpha) * hybrid_recs['cb_score']

    # Return top N recommendations sorted by the hybrid score
    return hybrid_recs.sort_values(by='hybrid_score', ascending=False).head(num_recommendations)
