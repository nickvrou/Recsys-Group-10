from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def calculate_popularity_score(df, weights=None):
    """
    Calculate a popularity score based on polarity, number of reviews, and review scores.
    """
    # Default weights if none are provided
    if weights is None:
        weights = {'polarity': 0.6, 'number_of_reviews': 0.1, 'review_scores_rating': 0.3}

    # Calculate recency by days since listing
    df['days_since_listing'] = (pd.Timestamp.today() - pd.to_datetime(df['date'], errors='coerce')).dt.days

    # Normalize recency: newer listings get higher scores
    scaler = MinMaxScaler()
    df['recency_score'] = 1 - scaler.fit_transform(df[['days_since_listing']].fillna(0))

    # Normalize relevant factors (polarity, number of reviews, review scores)
    popularity_factors = df[['polarity', 'number_of_reviews', 'review_scores_rating']]
    normalized_factors = pd.DataFrame(scaler.fit_transform(popularity_factors),
                                      columns=popularity_factors.columns, index=df.index)

    # Calculate weighted popularity score
    df['popularity_score'] = (weights['polarity'] * normalized_factors['polarity'] +
                              weights['number_of_reviews'] * normalized_factors['number_of_reviews'] +
                              weights['review_scores_rating'] * normalized_factors['review_scores_rating'])

    return df
