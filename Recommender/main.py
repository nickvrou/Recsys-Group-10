from sklearn.metrics.pairwise import cosine_similarity
from content_based import build_combined_features, get_content_based_recommendations, get_listing_index, \
    get_mmr_recommendations
from data_loader import load_data
from popularity_filtering import calculate_popularity_score


def run_content_based_recommender(df_grouped, listing_id, num_recommendations=5):
    """
    Runs the content-based recommendation system and returns the recommendations.
    """
    combined_features, group_df = build_combined_features(df_grouped)
    recommendations = get_content_based_recommendations(listing_id, group_df, combined_features, num_recommendations)

    return recommendations, combined_features, group_df

def run_mmr_recommender(df, listing_id, num_recommendations=5, lambda_param=0.7):
    """
    Runs the MMR (Maximal Marginal Relevance) recommendation system and returns the recommendations.
    """
    combined_features, df_grouped = build_combined_features(df)

    recommendations = get_mmr_recommendations(listing_id, df_grouped, combined_features, lambda_param, num_recommendations)

    return recommendations, combined_features, df_grouped

def run_recommender_pipeline():
    """
    Main function to run all recommenders (content-based, SVD, and hybrid).
    """
    # Load preprocessed data
    df = load_data('C:/Users/astratou/Downloads/final_preprocessed_df (3).csv')

    listing_id = '2818'

    # --- Run content-based recommender ---
    content_based_recs, combined_features, group_df = run_content_based_recommender(df, listing_id,num_recommendations=5)
    content_based_recs['cb_score'] = cosine_similarity(
        combined_features[get_listing_index(listing_id, group_df)],
        combined_features[content_based_recs.index.tolist()]).flatten()

    # Apply popularity filtering to the content-based recommendations
    if not content_based_recs.empty:
        content_based_recs_with_popularity = calculate_popularity_score(content_based_recs)  # Apply popularity score
        # Sort content-based recommendations by popularity score
        sorted_content_recs = content_based_recs_with_popularity.sort_values(by='popularity_score', ascending=False).head(5)

        print("Content-Based Recommendations Sorted by Popularity:")
        print(sorted_content_recs[['listing_id', 'cb_score', 'popularity_score']])
    else:
        print("No content-based recommendations found.")

    # --- Run MMR-based recommender ---
    mmr_recs, combined_features_mmr, group_df_mmr = run_mmr_recommender(df, listing_id, num_recommendations=5,
                                                                        lambda_param=0.6)

    mmr_recs['mmr_score'] = cosine_similarity(
        combined_features_mmr[get_listing_index(listing_id, group_df_mmr)],
        combined_features_mmr[mmr_recs.index.tolist()]).flatten()

    # Apply popularity filtering to the MMR recommendations
    if not mmr_recs.empty:
        mmr_recs_with_popularity = calculate_popularity_score(mmr_recs)  # Apply popularity score
        # Sort MMR recommendations by popularity score
        sorted_mmr_recs = mmr_recs_with_popularity.sort_values(by='popularity_score', ascending=False).head(5)

        print("\nMMR Recommendations Sorted by Popularity:")
        print(sorted_mmr_recs[['listing_id', 'mmr_score', 'popularity_score']])
    else:
        print("No MMR recommendations found.")

if __name__ == "__main__":
    run_recommender_pipeline()
