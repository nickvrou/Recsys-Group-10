from recommender.cf_svd import recommend_predictions, get_svd_recommendations
from recommender.content_based import build_combined_features, get_content_based_recommendations
from recommender.hybrid_recsys import get_hybrid_recommendations
from recommender.data_loader import load_data

def run_content_based_recommender(df, listing_id, num_recommendations=5):
    """
    Runs the content-based recommendation system and prints the recommendations.
    :param df: The input dataframe with listings.
    :param listing_id: The listing to base recommendations on.
    :param num_recommendations: Number of recommendations to return.
    """
    # Build content-based features
    df_grouped = df.groupby('listing_id').first().reset_index()
    combined_features = build_combined_features(df_grouped)

    # Get content-based recommendations
    recommendations = get_content_based_recommendations(listing_id, df_grouped, combined_features, num_recommendations)

    print("\nContent-Based Recommendations:")
    print(recommendations[['listing_id', 'name', 'description']])


def run_svd_recommender(df, reviewer_id, num_recommendations=5):
    """
    Runs the SVD-based recommendation system and prints the recommendations.
    :param df: The input dataframe with reviewer-item interactions.
    :param reviewer_id: The user to get recommendations for.
    :param num_recommendations: Number of recommendations to return.
    """
    # Run SVD to get predictions
    k = 50  # Number of latent features for SVD
    predictions, users_index, items_index = recommend_predictions(df, k, 'reviewer_id', 'listing_id', 'polarity')

    # Get SVD-based recommendations
    recommendations = get_svd_recommendations(predictions, reviewer_id, users_index, items_index,
                                              df['listing_id'].unique(), num_recommendations)

    print("\nSVD-Based (Collaborative Filtering) Recommendations:")
    print(recommendations[['listing_id', 'predicted_polarity']])


def run_hybrid_recommender(df, listing_id, reviewer_id, num_recommendations=5, alpha=0.5):
    """
    Runs the hybrid recommendation system and prints the recommendations.
    :param df: The input dataframe with listings and reviewer-item interactions.
    :param listing_id: The listing to base content-based recommendations on.
    :param reviewer_id: The reviewer to base SVD recommendations on.
    :param num_recommendations: Number of recommendations to return.
    :param alpha: The weight to combine content-based and SVD recommendations.
    """
    # Build content-based features
    df_grouped = df.groupby('listing_id').first().reset_index()
    combined_features = build_combined_features(df_grouped)

    # Run SVD to get predictions
    k = 50  # Number of latent features for SVD
    predictions, users_index, items_index = recommend_predictions(df, k, 'reviewer_id', 'listing_id', 'polarity')

    # Get hybrid recommendations
    hybrid_recommendations = get_hybrid_recommendations(listing_id, reviewer_id, df_grouped, combined_features,
                                                        predictions, users_index, items_index, alpha,
                                                        num_recommendations)

    print("\nHybrid Recommendations (Content-Based + SVD):")
    print(hybrid_recommendations[['listing_id', 'hybrid_score']])


def run_recommender_pipeline():
    """
    Main function to run all recommenders (content-based, SVD, and hybrid).
    """
    # Load preprocessed data
    df = load_data('preprocessed_listings.csv')

    # Define example listing_id and reviewer_id for recommendations
    listing_id = 'synthetic_001'  # Example listing_id for content-based
    reviewer_id = df['reviewer_id'].iloc[0]  # Example reviewer_id for SVD

    # Run Content-Based Recommender
    run_content_based_recommender(df, listing_id, num_recommendations=5)

    # Run SVD Recommender (Collaborative Filtering)
    run_svd_recommender(df, reviewer_id, num_recommendations=5)

    # Run Hybrid Recommender (Content-Based + SVD)
    run_hybrid_recommender(df, listing_id, reviewer_id, num_recommendations=5, alpha=0.5)


if __name__ == "__main__":
    run_recommender_pipeline()
