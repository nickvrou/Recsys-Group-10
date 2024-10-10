import pandas as pd

def applyLeastMisery(group_matrix):
    least_misery_scores = group_matrix.min() #min of each column
    top5 = least_misery_scores.sort_values(ascending=False).head(5) 
    dict = {}
    for listing_id, score in top5.items():
        dict[listing_id] = score
    return dict #return a dictionaries mapping top5 listing_ids to their scores
    
    
def applyMostPleasure(group_matrix):
    most_pleasure_scores = group_matrix.max()
    top5 = most_pleasure_scores.sort_values(ascending=False).head(5)
    dict = {}
    for listing_id, score in top5.items():
        dict[listing_id] = score
    return dict


def applyAverageStrategy(group_matrix):
    column_sums = group_matrix.sum()
    top5 = column_sums.sort_values(ascending=False).head(5)
    dict = {}
    for listing_id, score in top5.items():
        dict[listing_id] = score
    return dict
    
#method to get the ratings matrix to then apply a group strategy
def computeRatingsMatrix(content_based_recs_list, listings_id):
    ratings_matrix = pd.DataFrame()
    for i in range(len(content_based_recs_list)): # loop throug the 4 users recommendations
        for key, value in content_based_recs_list[i].items():
            if str(key) not in listings_id:  # removed the 4 users
                ratings_matrix.at[listings_id[i], key] = value  #adding the value to each user-listing postion 
          

    return ratings_matrix 