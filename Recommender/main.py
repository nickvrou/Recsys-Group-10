from sklearn.metrics.pairwise import cosine_similarity
from cf_svd import recommend_predictions, get_svd_recommendations
from content_based import build_combined_features, get_content_based_recommendations, get_listing_index
from data_loader import load_data
from group_recommender import applyLeastMisery, applyMostPleasure, applyAverageStrategy,computeRatingsMatrix
import pandas as pd

import warnings
warnings.filterwarnings("ignore")



def run_content_based_recommender(df_grouped, listing_id,num_recommendations=5):
    """
    Runs the content-based recommendation system and returns the recommendations.
    """
    combined_features, group_df = build_combined_features(df_grouped)
    recommendations = get_content_based_recommendations(listing_id, group_df, combined_features, num_recommendations)
    print('Done content')
    return recommendations, combined_features, group_df
 


def run_recommender_pipeline():
    """
    Main function to run all recommenders (content-based, SVD, and hybrid).
    """
    # Load preprocessed data
    df = load_data('recommender/final_preprocessed_df (3).zip')

    listings_id = ['742160','307621','2818','327285'] #user preferences (specific listings)
    
    #Run Content-Based Recommender
    content_based_recs_list = []
    all_listing_ids = df['listing_id'].unique()
    print(len(all_listing_ids))

    #Iterating through each user and getting the recommendations
    for listing_id in listings_id:
        content_based_recs, combined_features, group_df = run_content_based_recommender(df, listing_id, num_recommendations=len(all_listing_ids))
        content_based_recs['cb_score'] = cosine_similarity(
            combined_features[get_listing_index(listing_id, group_df)],
            combined_features[content_based_recs.index]
        ).flatten()

        # make a dicitionary mapping listing_ids to cb_scores 
        dictionary = dict(zip(content_based_recs['listing_id'], content_based_recs['cb_score']))     
        content_based_recs_list.append(dictionary)  #do the same for each user and store it in a list



    #getting the ratings matrix to apply group recommendation strategies    
    ratings_matrix=computeRatingsMatrix(content_based_recs_list, all_listing_ids)


    print(applyLeastMisery(ratings_matrix))
    # applyMostPleasure(ratings_matrix)
    # applyAverageStrategy(ratings_matrix)

if __name__ == "__main__":
    run_recommender_pipeline()
