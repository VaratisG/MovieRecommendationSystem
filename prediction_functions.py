import numpy as np

def predict_weighted_average(user_id, item_id, train_sparse_matrix, movie_id_to_index, user_id_to_index, similarity_matrix, N):
    """
    Predict rating using similarity-weighted average of top N neighbors.
    Returns NaN if the movie or user is not in the training set.
    """
    if item_id not in movie_id_to_index or user_id not in user_id_to_index:
        return np.nan  # Handle missing movies/users gracefully

    user_idx = user_id_to_index[user_id]
    item_idx = movie_id_to_index[item_id]

    user_ratings = train_sparse_matrix[user_idx, :].toarray().flatten()
    rated_indices = np.where(user_ratings > 0)[0]

    similarities = similarity_matrix[item_idx, rated_indices]
    top_indices = np.argsort(-similarities)[:N]
    top_similarities = similarities[top_indices]
    top_ratings = user_ratings[rated_indices[top_indices]]

    return np.dot(top_ratings, top_similarities) / np.sum(top_similarities) if np.sum(top_similarities) > 0 else np.nan

def predict_weighted_popularity(user_id, item_id, train_sparse_matrix, movie_ids, movie_id_to_index, user_id_to_index, similarity_matrix, item_popularity, favor_popular, N):
    """
    Predict rating using popularity-adjusted similarity weights.
    Returns NaN if the movie or user is not in the training set.
    """
    if item_id not in movie_id_to_index or user_id not in user_id_to_index:
        return np.nan  # Handle missing movies/users gracefully

    user_idx = user_id_to_index[user_id]
    item_idx = movie_id_to_index[item_id]

    user_ratings = train_sparse_matrix[user_idx, :].toarray().flatten()
    rated_indices = np.where(user_ratings > 0)[0]

    similarities = similarity_matrix[item_idx, rated_indices]
    top_indices = np.argsort(-similarities)[:N]
    top_similarities = similarities[top_indices]
    top_ratings = user_ratings[rated_indices[top_indices]]
    top_popularities = np.array([item_popularity.get(movie_ids[idx], 0) for idx in rated_indices[top_indices]])

    weights = top_similarities * (1 + np.log1p(top_popularities)) if favor_popular else \
              top_similarities / (1 + np.log1p(top_popularities))

    return np.dot(top_ratings, weights) / np.sum(weights) if np.sum(weights) > 0 else np.nan
