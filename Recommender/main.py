from datetime import date
import re
from sklearn.metrics.pairwise import cosine_similarity
from content_based import build_combined_features, get_two_stage_recommendations
from data_loader import load_data
from group_recommender import applyLeastMisery, computeRatingsMatrix
import pandas as pd
import warnings
import gensim

from explainations import explain_recommendation_with_word2vec
warnings.filterwarnings("ignore")


word2vec_path = 'recommender/GoogleNews-vectors-negative300.bin.gz'
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)


def run_recommender_pipeline():

    def get_listing_index(listing_id, df_grouped):

        listing_id = str(listing_id).strip()
        
        df_grouped['listing_id'] = df_grouped['listing_id'].astype(str).str.strip()

        try:
            return df_grouped[df_grouped['listing_id'] == listing_id].index[0]
        except IndexError:  
            return None
        
    """
    Main function to run group recommender.
    """
    # Load preprocessed data
    df = load_data('recommender/final_preprocessed_df (3).zip')

    user_ids = ['1','2','3','4'] #user preferences (specific listings)

    #Run Content-Based Recommender

    # ask user for constraints
    max_price = int(input('Enter the maximum price: '))
    min_price = int(input('Enter the minimum price: '))
    min_accommodates = int(input('Enter the minimum number of accommodates: '))


    def filter_listings_by_constraints(df, max_price, min_price, min_accommodates):
        # Filter based on price and accommodates
        filtered_df = df[(df['price'] >= min_price) & 
                        (df['price'] <= max_price) & 
                        (df['accommodates'] >= min_accommodates)]
        return filtered_df
        
    # apply constraint based part 
    df = filter_listings_by_constraints(df, max_price, min_price, min_accommodates) 


    content_based_recs_list = []     #store the recommendations for all 4 users
    all_listing_ids = df['listing_id'].unique()
    description_list= ''
    
    for i in range(4): 
        print(f"User {i+1}")
                
        available_amenities = ["Wi-Fi", "Kitchen", "Parking", "Pet-friendly", 
                       "Air conditioning", "Smoke alarm", "Heating", 
                       "Iron", "BBQ grill", "Bikes"]
        print(f"Available amenities: {', '.join(available_amenities)}")
        amenities = input("Enter the amenities you want , separate them by commas): ")
        amenities_list = amenities.split(",") 
        description= input('Enter the description: ')

        description_list+= description

        user_input={
            'listing_id': user_ids[i],
            'price': (min_price + max_price) / 2,  
            'property_type': ' ',  
            'amenities': ', '.join(amenities_list),  
            'comments': description, 
            'description': description,
            'name': 'abc',  
            'bathrooms': 1,
            'bedrooms': 2,
            'beds': 4,  
            'minimum_nights': (date(2024,10,6) - date(2024,10,6)).days,
            'distance_to_center': (5 + 10) / 2,
            'accommodates': 4
        }

        df= pd.concat([df, pd.DataFrame([user_input])], ignore_index=True)

        df['listing_id'] = df['listing_id'].astype(str).str.strip()

        df[df['listing_id'] == 'user_input'].head()


    for user_id in user_ids:

        #Content based application
        textual_embeddings, structured_features, df_grouped = build_combined_features(df)
        
        recommendations = get_two_stage_recommendations(user_id, df_grouped, textual_embeddings, structured_features,num_final_recommendations=len(all_listing_ids))

        dictionary = dict(zip(recommendations['listing_id'],  recommendations['final_similarity_score']))  #make a dicitionary mapping listing_ids to similarity scores 

        content_based_recs_list.append(dictionary)  #do the same for each user and store it in a list

    #getting the ratings matrix to apply group recommendation strategies    
    ratings_matrix = computeRatingsMatrix(content_based_recs_list, user_ids)

    #least misery strategy
    dict_recommendations = applyLeastMisery(ratings_matrix) #dictionary mapping the best listing_ids to their scores


  

    stop_words = [
    'the', 'is', 'in', 'and', 'to', 'a', 'of', 'it', 'we', 'was', 'for', 'It',
    'but', 'on', 'with', 'as', 'you', 'at', 'this', 'that', 'had', 'our',
    'be', 'by', 'or', 'an', 'are', 'from', 'so', 'if', 'have', 'my',
    'they', 'which', 'one', 'their', 'there', 'what', 'more', 'when',
    'can', 'your', 'will', 'would', 'should', 'could', 'about', 'out', 'up',
    'them', 'some', 'me', 'just', 'into', 'has', 'also', 'very', 'been',
    'did', 'do', 'he', 'she', 'his', 'her', 'how', 'then', 'than', 'other',
    'over', 'because', 'any', 'only', 'were', 'after', 'did', 'these',
    'who', 'its', 'see', 'well', 'here', 'get', 'got', 'even', 'make',
    'made', 'us', 'you', 'your', 'yours', 'I', 'am', 'he', 'she', 'it',
    'we', 'they', 'you', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'myself', 'ourselves', 'yourself', 'yourselves',
    'himself', 'herself', 'itself', 'themselves', 'each', 'few', 'many',
    'some', 'such', 'very', 's', 't', 'can', 'will', 'don', 'should', 'now']
                                             
    def remove_stop_words(text, stop_words):
    
        if pd.isnull(text):
            return text

        # Remove punctuation and non-alphabetical characters
        text = re.sub(r'[^\w\s]', '', text)

        # Split text into words
        words = text.split()

        # Filter out stop words
        filtered_words = [word for word in words if word.lower() not in stop_words]

        # Rejoin the filtered words into a cleaned sentence
        filtered_text = ' '.join(filtered_words)

        return filtered_text                              

    
    recommendation_df = df[df['listing_id'].isin(dict_recommendations.keys())] #get the listings that are recommended by the group strategy in a dataframe

    recommendation_df = recommendation_df.drop_duplicates(subset=['listing_id'])


    # Setting the maximum column width for displaying dataframes
    pd.set_option('display.max_colwidth',None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # print(recommendation_df[['name','description','review_scores_rating','bathrooms','bedrooms','distance_to_center','property_type','price','accommodates']].head())

    recommendation_df.to_csv('recommender_df.csv', index=False)

    
    
    #Group explanation using word2vec
    for _, rec in recommendation_df.iterrows():
                filtered_comments = remove_stop_words(rec['comments'], stop_words)
                filtered_description = remove_stop_words(rec['description'], stop_words)
                filtered_description = remove_stop_words(description_list, stop_words)
                explanation = explain_recommendation_with_word2vec(
                    description_list,
                    filtered_description,  # The user's input (after removing stop words)
                    filtered_comments,
                    word2vec_model)  # Word2Vec model
                
                print(f"We recommend '{rec['name']}' because {explanation}")    

if __name__ == "__main__":
    run_recommender_pipeline()
