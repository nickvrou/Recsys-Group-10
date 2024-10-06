import pandas as pd
def applyAverageStrategy(group_matrix):
    # Sum each column in the matrix and create a dictionary where keys are column names (listing_ids) and values are the sums
    column_sums = {}

    # Iterate over each column in the DataFrame
    for listing_id in group_matrix.columns:
        # Sum the values of the column and store in the dictionary
        column_sums[listing_id] = group_matrix[listing_id].sum()

    # choose the biggest 10
    top_10 = sorted(column_sums.items(), key=lambda item: item[1], reverse=True)[:10]

    # Print the lowest 10 listings with their sums
    for key, value in top_10:
        print(f"Listing ID: {key}, Sum: {value}")

def applyLeastMisery(group_matrix):
    # Find the maximum score for each listing
    most_pleasure_scores = group_matrix.min()

    # Sort the items based on most pleasure score in descending order
    top_10_most_pleasure = most_pleasure_scores.sort_values(ascending=False).head(10)

    # Print the top 10 listings with the highest most pleasure scores
    print("Top 10 listings based on Most Pleasure strategy:")
    for listing_id, score in top_10_most_pleasure.items():
        print(f"Listing ID: {listing_id}, Most Pleasure Score: {score}")
    
    
def applyMostPleasure(group_matrix):
    # Find the maximum score for each listing
    most_pleasure_scores = group_matrix.max()

    # Sort the items based on most pleasure score in descending order
    top_10_most_pleasure = most_pleasure_scores.sort_values(ascending=False).head(10)

    # Print the top 10 listings with the highest most pleasure scores
    print("Top 10 listings based on Most Pleasure strategy:")
    for listing_id, score in top_10_most_pleasure.items():
        print(f"Listing ID: {listing_id}, Most Pleasure Score: {score}")


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