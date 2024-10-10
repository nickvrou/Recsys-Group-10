# Airbnb Recommender System

This project implements an **Airbnb Recommender System** using a combination of content-based filtering and popularity-based recommendations. The system is built using Python and leverages the Word2Vec model to calculate semantic similarity between listings based on textual data such as descriptions, comments, and amenities.

## Features

- **Content-Based Recommendations**: Generates recommendations by analyzing the metadata (description, amenities, reviews, etc.) of listings using Word2Vec embeddings for textual data.
- **Popularity-Based Recommendations**: Sorts listings by a calculated popularity score based on user reviews, rating polarity, and the number of reviews.
- **Word2Vec Semantic Similarity**: Uses pre-trained Google News Word2Vec vectors to assess the similarity between listings and user input.
- **Streamlit GUI**: A simple user interface for users to input search criteria and retrieve recommendations.

## Repository Structure

Recommender/
│
├── __pycache__/           
├── __init__.py            # Initialization for the Recommender package (used for group recommender in teh beginning)
├── app.py                 # Streamlit app for the Airbnb recommender
├── content_based.py       # Content-based recommendation logic
├── data_loader.py          # Data loading and preprocessing functionality
├── explainations.py        # Word2Vec-based explanation functions
├── group_recommender.py    # Helper functions for group recommedner in main.py
├── main.py                 # Group-recommender activation-logic code
├── popularity_filtering.py # Popularity-based scoring and recommendation filtering
├── requirements.txt        # Required Python packages and dependencies
├── original_dataset.zip    # Original Airbnb dataset  - not preprocessed (for recommendations)
├── final_preprocessed (3).zip  #Preprocessed


## Usage

### Individual Recommender

1) Clone directory
2) Install required packages (pip install -r requirements.txt)
3) Download the pre-trained Word2Vec model (open-source online)
4) Run the app using Streamlit (streamlit run Recommender/app.py)

### GUI

Airbnb Recommender System: Enter your search criteria (location, number of guests, price range, etc.) and click "Get Recommendations." The system will generate a list of recommended listings, along with explanations for why each listing was recommended.
The system supports filtering based on the number of guests and price range

### Group Recommender

1) Clone directory
2) Install required packages (pip install -r requirements.txt)
3) Download the pre-trained Word2Vec model (open source online)
4) Run the main.py and get recommendations and explanations

## Online user survey

Link for the survey: https://qualtricsxmcrj5srxjl.qualtrics.com/jfe/form/SV_esoFBJUz9qFqW58
