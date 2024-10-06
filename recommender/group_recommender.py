import pandas as pd

def applyLeastMisery(group_matrix):
    most_pleasure_scores = group_matrix.min()
    top_10 = most_pleasure_scores.sort_values(ascending=False).head(10)
    dict = {}
    for listing_id, score in top_10.items():
        dict[listing_id] = score
    return dict
    
    
def applyMostPleasure(group_matrix):
    most_pleasure_scores = group_matrix.max()
    top_10 = most_pleasure_scores.sort_values(ascending=False).head(10)
    dict = {}
    for listing_id, score in top_10.items():
        dict[listing_id] = score
    return dict


def applyAverageStrategy(group_matrix):
    column_sums = group_matrix.sum()
    top_10 = column_sums.sort_values(ascending=False).head(10)
    dict = {}
    for listing_id, score in top_10.items():
        dict[listing_id] = score
    return dict
    

def computeRatingsMatrix(content_based_recs_list, listings_id):
    ratings_matrix = pd.DataFrame()
    # loop through the content_based_recs_list and get the listing_id and cb_score
    for i in range(len(content_based_recs_list)):
        for key, value in content_based_recs_list[i].items():
            if str(key) not in listings_id:  # removed the 4 users
                ratings_matrix.at[listings_id[i], key] = value
            else:
                continue

    return ratings_matrix