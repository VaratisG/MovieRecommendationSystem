import time
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from multiprocessing import Pool, shared_memory

from prediction_functions import(
    predict_weighted_average,
    predict_weighted_popularity
)

def print_progress(current, total, start_time):
    """
    Display progress percentage, elapsed time and estimated remaining time
    Args:
        current: Current item being processed
        total: Total number of items to process
        start_time: Timestamp when processing started
    """
    elapsed = time.time() - start_time
    percent = (current / total) * 100
    remaining = (elapsed / (current + 1)) * (total - current) if current > 0 else 0
    print(f"\rProgress: {percent:.1f}% | Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s", end="")

def train_test_split_ratings(ratings, test_size=0.2, random_state=42):
    """
    Split ratings into training and testing sets, ensuring test set only contains
    movies and users present in the training set.
    """
    # Split the data
    train, test = train_test_split(ratings, test_size=test_size, random_state=random_state)
    
    # Filter test set to only include movies and users present in the training set
    train_movies = set(train['movieId'])
    train_users = set(train['userId'])
    
    test = test[test['movieId'].isin(train_movies) & test['userId'].isin(train_users)]
    
    return train, test

def filter_ratings(M, M_prime):
    """
    Filter the ratings dataset to include only movies with at least M ratings
    and users with at least M' ratings.
    Args:
        M: Minimum number of ratings per movie
        M_prime: Minimum number of ratings per user
    Returns:
        Filtered DataFrame
    """
    # Load the ratings dataset
    ratings = pd.read_csv("csv/ratings.csv")

    # Filter movies with at least M ratings
    movie_counts = ratings['movieId'].value_counts()
    filtered_movies = movie_counts[movie_counts >= M].index
    ratings = ratings[ratings['movieId'].isin(filtered_movies)]

    # Filter users with at least M' ratings
    user_counts = ratings['userId'].value_counts()
    filtered_users = user_counts[user_counts >= M_prime].index
    ratings = ratings[ratings['userId'].isin(filtered_users)]

    # Print summary
    print(f"Filtered Dataset (M={M}, M'={M_prime}):")
    print(f"- Unique users: {ratings['userId'].nunique()}")
    print(f"- Unique movies: {ratings['movieId'].nunique()}")
    print(f"- Total ratings: {len(ratings)}")

    return ratings

def create_sparse_matrix(ratings_df):
    """
    Create sparse user-item matrix from ratings data
    Args:
        ratings_df: DataFrame containing user ratings
    Returns:
        Tuple of (sparse_matrix, movie_ids, user_ids)
    """
    user_item_matrix = ratings_df.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)
    sparse_matrix = csr_matrix(user_item_matrix.values)
    return sparse_matrix, list(user_item_matrix.columns), list(user_item_matrix.index)

def compute_similarity(sparse_matrix, method='cosine'):
    """
    Compute item-item similarity matrix using specified method
    Args:
        sparse_matrix: Sparse user-item matrix
        method: 'cosine' for cosine similarity, 'pearson' for Pearson correlation
    Returns:
        Dense similarity matrix
    """
    if method == 'cosine':
        return cosine_similarity(sparse_matrix.T, dense_output=True)
    elif method == 'pearson':
        # Pearson correlation implementation using sparse matrix operations
        user_means = sparse_matrix.mean(axis=1).A1
        centered_matrix = sparse_matrix.copy()
        centered_matrix.data -= user_means[sparse_matrix.nonzero()[0]]
        cov = centered_matrix.T @ centered_matrix
        std_dev = np.sqrt(np.array(cov.diagonal()).clip(min=1e-9))
        pearson = cov / np.outer(std_dev, std_dev)
        return pearson.toarray()
    else:
        raise ValueError("Invalid similarity method. Choose 'cosine' or 'pearson'.")

def predict(row, train_sparse_matrix, movie_ids, movie_id_to_index, user_id_to_index, similarity_matrix, item_popularity, favor_popular, N):
    """
    Route prediction to appropriate function based on weighting type
    Args:
        row: Row from test DataFrame containing userId and movieId
        Other args: Model parameters and data structures
    Returns:
        Predicted rating for the user-item pair
    """
    user_id = row['userId']
    item_id = row['movieId']
    if favor_popular is None:
        return predict_weighted_average(user_id, item_id, train_sparse_matrix, movie_id_to_index, user_id_to_index, similarity_matrix, N)
    else:
        return predict_weighted_popularity(user_id, item_id, train_sparse_matrix, movie_ids, movie_id_to_index, user_id_to_index, similarity_matrix, item_popularity, favor_popular, N)

def parallel_prediction(test_ratings, train_sparse_matrix, movie_ids, user_ids, similarity_matrix, item_popularity, favor_popular, N):
    """
    Perform parallel prediction using shared memory for similarity matrix
    Args:
        test_ratings: DataFrame of test ratings to predict
        Other args: Model parameters and data structures
    Returns:
        List of predicted ratings
    """
    total_items = len(test_ratings)
    start_time = time.time()
    
    # Create lookup dictionaries for O(1) index access
    movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}

    # Set up shared memory for similarity matrix
    shm = shared_memory.SharedMemory(create=True, size=similarity_matrix.nbytes)
    shared_sim = np.ndarray(similarity_matrix.shape, dtype=similarity_matrix.dtype, buffer=shm.buf)
    np.copyto(shared_sim, similarity_matrix)

    print(f"\nStarting predictions for N={N}...")
    
    # Process predictions in parallel using 4 workers
    with Pool(processes=4) as pool:
        predictions = pool.starmap(
            predict,
            [(row, train_sparse_matrix, movie_ids, movie_id_to_index, user_id_to_index, shared_sim, 
              item_popularity, favor_popular, N) 
             for _, row in test_ratings.iterrows()]
        )
    
    # Clean up shared memory
    shm.close()
    shm.unlink()
    
    print_progress(total_items, total_items, start_time)
    print(f"\nPredictions completed for N={N} in {time.time()-start_time:.1f}s")
    return predictions

def calculate_metrics(test_ratings, predicted_ratings):
    """
    Calculate evaluation metrics (MAE, Precision, Recall)
    Args:
        test_ratings: Ground truth ratings from test set
        predicted_ratings: Model predictions
    Returns:
        Tuple of (MAE, precision, recall)
    """
    test_ratings = test_ratings.reset_index(drop=True)
    # Calculate Mean Absolute Error
    mae = np.nanmean(np.abs(test_ratings['rating'] - predicted_ratings))

    # Initialize confusion matrix components
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Calculate precision and recall using 3.5 rating threshold
    for i, pred in enumerate(predicted_ratings):
        if np.isnan(pred):
            continue
        actual = test_ratings.at[i, 'rating']
        if actual >= 3.5 and pred >= 3.5:
            true_positives += 1
        elif actual < 3.5 and pred >= 3.5:
            false_positives += 1
        elif actual >= 3.5 and pred < 3.5:
            false_negatives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return mae, precision, recall