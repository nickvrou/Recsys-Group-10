from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def calculate_popularity_score(df, weights=None):
    """
    Calculate a popularity score based on multiple factors and weights.
    :param df: DataFrame with the columns ['polarity', 'synthetic_rating', 'number_of_reviews', 'review_scores_rating']
    :param weights: Optional dictionary to assign weights to each factor
    :return: DataFrame with an additional 'popularity_score' column
    """
    # Set default weights if not provided
    if weights is None:

        weights = {

            'polarity': 0.6,

            'number_of_reviews': 0.1,

            'review_scores_rating': 0.3,

        }

    reference_date = pd.Timestamp.today()

    df['days_since_listing'] = (reference_date - pd.to_datetime(df['date'], errors='coerce')).dt.days

    # Convert recency to a normalized score (lower days since = more recent = higher recency score)
    scaler = MinMaxScaler()

    df['recency_score'] = 1 - scaler.fit_transform(df[['days_since_listing']].fillna(0))

    # Select only the relevant columns for popularity scoring, including recency
    popularity_factors = df[['polarity', 'number_of_reviews', 'review_scores_rating']]

    # Normalize the factors using MinMaxScaler to scale them between 0 and 1
    normalized_factors = scaler.fit_transform(popularity_factors)

    normalized_df = pd.DataFrame(normalized_factors, columns=popularity_factors.columns, index=df.index)

    # Calculate the weighted popularity score
    df['popularity_score'] = (

        weights['polarity'] * normalized_df['polarity'] +

        weights['number_of_reviews'] * normalized_df['number_of_reviews'] +

        weights['review_scores_rating'] * normalized_df['review_scores_rating']

    )

    return df