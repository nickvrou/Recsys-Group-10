from sklearn.metrics.pairwise import cosine_similarity
from cf_svd import recommend_predictions, get_svd_recommendations
from content_based import build_combined_features, get_content_based_recommendations, get_listing_index
from data_loader import load_data
import pandas as pd


def run_content_based_recommender(df_grouped, listing_id, num_recommendations=5):
    """
    Runs the content-based recommendation system and returns the recommendations.
    """
    combined_features, group_df = build_combined_features(df_grouped)
    recommendations = get_content_based_recommendations(listing_id, group_df, combined_features, num_recommendations)
    print('Done content')
    return recommendations, combined_features, group_df


# def run_svd_recommender(df, reviewer_id, num_recommendations=5, k=50):
#     """
#     Runs the SVD-based recommendation system and returns the recommendations.
#     """
#     predictions, users_index, items_index = recommend_predictions(df, k, 'reviewer_id', 'listing_id', 'polarity')
#     recommendations = get_svd_recommendations(predictions, reviewer_id, users_index, items_index,
#                                               df['listing_id'].unique(), num_recommendations)
#     print('Done svd')
#     return recommendations, predictions, users_index, items_index


# def run_hybrid_recommender(listing_id, reviewer_id, content_based_recs, combined_features, group_df,
#                            svd_recs, predictions, users_index, items_index, alpha=0.5, num_recommendations=5):
#     """
#     Runs the hybrid recommendation system using pre-calculated content-based and SVD recommendations.
#     """
#     # Merge the recommendations based on listing_id
#     hybrid_recs = pd.merge(content_based_recs[['listing_id', 'cb_score']],
#                            svd_recs[['listing_id', 'predicted_polarity']], on='listing_id', how='inner')

#     # Normalize the collaborative filtering score (predicted polarity)
#     hybrid_recs['cf_score'] = hybrid_recs['predicted_polarity'] / hybrid_recs['predicted_polarity'].max()

#     # Calculate the hybrid score (weighted combination)
#     hybrid_recs['hybrid_score'] = alpha * hybrid_recs['cf_score'] + (1 - alpha) * hybrid_recs['cb_score']

#     # Return top N recommendations sorted by the hybrid score
#     return hybrid_recs.sort_values(by='hybrid_score', ascending=False).head(num_recommendations)


def run_recommender_pipeline():
    """
    Main function to run all recommenders (content-based, SVD, and hybrid).
    """
    # Load preprocessed data
    df = load_data('C:/Users/vroun\Documents/GitHub/Recsys-Group-10/recommender/final_preprocessed_df (3).csv')

    listing_id = {
        'comments': None,
        'name': None,
        'description': None,
        'property_type': None,
        'review_scores_rating': None,
        'bathrooms': None,
        'bedrooms': None,
        'beds': None,
        'minimum_nights': None,
        'maximum_nights': None,
        'distance_to_center': None,
        'polarity': None,
        'synthetic_rating': None,
        'amenities': None,
        'number_of_reviews':None,
        'date': None
    }
    reviewer_id = df['reviewer_id'].iloc[0]

    # 1. Run Content-Based Recommender
    content_based_recs, combined_features, group_df = run_content_based_recommender(df, listing_id, num_recommendations=5)
    content_based_recs['cb_score'] = cosine_similarity(
        combined_features[get_listing_index(listing_id, group_df)],
        combined_features[content_based_recs.index]).flatten()

    # # 2. Run SVD Recommender (Collaborative Filtering)
    # svd_recs, predictions, users_index, items_index = run_svd_recommender(df, reviewer_id, num_recommendations=5)

    # # 3. Run Hybrid Recommender (Content-Based + SVD)
    # hybrid_recs = run_hybrid_recommender(listing_id, reviewer_id, content_based_recs, combined_features, group_df,
    #                                      svd_recs, predictions, users_index, items_index, alpha=0.5, num_recommendations=5)

    # # Display recommendations
    # print("\nHybrid Recommendations (Content-Based + SVD):")
    # print(hybrid_recs[['listing_id', 'hybrid_score']])


if __name__ == "__main__":
    run_recommender_pipeline()
