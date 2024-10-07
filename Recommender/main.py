import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from content_based import build_combined_features, get_content_based_recommendations, get_listing_index, get_mmr_recommendations
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


def explain_recommendation(query_listing, recommended_listing, vectorizer):
    """
    Generate a human-readable explanation for why a recommendation was made, focusing on words from the description.
    """
    explanations = {}

    # Calculate similarity for the description using cosine similarity
    sim_description = cosine_similarity(vectorizer.transform([query_listing['description']]),
                                        vectorizer.transform([recommended_listing['description']]))[0][0]

    # Explanation of word importance for the description
    query_desc_vec = vectorizer.transform([query_listing['description']])
    rec_desc_vec = vectorizer.transform([recommended_listing['description']])

    # Get feature (word) names
    feature_names = vectorizer.get_feature_names_out()

    # Compute word-level importance by subtracting the vectors
    word_importance = np.abs(query_desc_vec - rec_desc_vec).toarray().flatten()

    # Sort words by importance and display the top words
    important_words = sorted(zip(feature_names, word_importance), key=lambda x: x[1], reverse=True)[:5]
    explanations['Top Important Words'] = important_words

    # Generate a natural language explanation
    top_words_str = ', '.join([word for word, _ in important_words])

    explanation_text = f"We recommend '{recommended_listing['name']}' because it matches the words you provided in your description, particularly: {top_words_str}."

    return explanation_text


def run_recommender_pipeline():
    """
    Main function to run all recommenders (content-based, MMR), applying popularity filtering.
    """
    # Load preprocessed data
    df = load_data('C:/Users/astratou/Downloads/final_preprocessed_df (3).csv')

    # Load vectorizer for generating text explanations
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the text columns (comments, description, amenities)
    df_text = df[['comments', 'description', 'amenities']].fillna('')
    vectorizer.fit(df_text.apply(lambda x: ' '.join(x), axis=1))

    # Define listing ID for the recommendations
    listing_id = '1154997978682668114'

    # --- Run content-based recommender ---
    content_based_recs, combined_features, group_df = run_content_based_recommender(df, listing_id, num_recommendations=5)
    content_based_recs['cb_score'] = cosine_similarity(
        combined_features[get_listing_index(listing_id, group_df)],
        combined_features[content_based_recs.index.tolist()]).flatten()

    # Apply popularity filtering to the content-based recommendations
    if not content_based_recs.empty:
        content_based_recs_with_popularity = calculate_popularity_score(content_based_recs)  # Apply popularity score
        sorted_content_recs = content_based_recs_with_popularity.sort_values(by='popularity_score', ascending=False).head(5)

        print("Content-Based Recommendations Sorted by Popularity:")
        print(sorted_content_recs[['listing_id', 'cb_score', 'popularity_score']])

        # Get explanations for each recommendation in natural language
        print("\nExplanations for Recommendations:")
        query_listing = df[df['listing_id'] == listing_id].iloc[0].to_dict()

        for index, rec in sorted_content_recs.iterrows():
            recommended_listing = df[df['listing_id'] == rec['listing_id']].iloc[0].to_dict()
            explanation_text = explain_recommendation(query_listing, recommended_listing, vectorizer)
            print(f"{explanation_text}")
    else:
        print("No content-based recommendations found.")

    # --- Run MMR-based recommender ---
    mmr_recs, combined_features_mmr, group_df_mmr = run_mmr_recommender(df, listing_id, num_recommendations=5, lambda_param=0.6)

    mmr_recs['mmr_score'] = cosine_similarity(
        combined_features_mmr[get_listing_index(listing_id, group_df_mmr)],
        combined_features_mmr[mmr_recs.index.tolist()]).flatten()

    # Apply popularity filtering to the MMR recommendations
    if not mmr_recs.empty:

        mmr_recs_with_popularity = calculate_popularity_score(mmr_recs)  # Apply popularity score
        sorted_mmr_recs = mmr_recs_with_popularity.sort_values(by='popularity_score', ascending=False).head(5)

        print("\nMMR Recommendations Sorted by Popularity:")
        print(sorted_mmr_recs[['listing_id', 'mmr_score', 'popularity_score']])

        # Get explanations for each MMR recommendation in natural language
        print("\nExplanations for MMR Recommendations:")
        for index, rec in sorted_mmr_recs.iterrows():
            recommended_listing = df[df['listing_id'] == rec['listing_id']].iloc[0].to_dict()
            explanation_text = explain_recommendation(query_listing, recommended_listing, vectorizer)
            print(f"{explanation_text}")
    else:
        print("No MMR recommendations found.")


if __name__ == "__main__":
    run_recommender_pipeline()
