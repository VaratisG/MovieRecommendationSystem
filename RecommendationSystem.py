import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from multiprocessing import Pool, shared_memory
import time
import warnings
from tabulate import tabulate
import sys

warnings.filterwarnings('ignore')

# Load movie ratings dataset from CSV file
ratings = pd.read_csv("csv/ratings.csv")

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
    Split ratings dataset into training and testing sets
    Args:
        ratings: DataFrame containing user ratings
        test_size: Proportion of dataset to allocate for testing
        random_state: Seed for reproducible splits
    Returns:
        Tuple of (train_ratings, test_ratings) DataFrames
    """
    return train_test_split(ratings, test_size=test_size, random_state=random_state)

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

def predict_weighted_average(user_id, item_id, train_sparse_matrix, movie_id_to_index, user_id_to_index, similarity_matrix, N):
    """
    Predict rating using similarity-weighted average of top N neighbors
    Args:
        user_id: Target user ID
        item_id: Target movie ID
        Other args: Model parameters and data structures
    Returns:
        Predicted rating or NaN if no valid neighbors
    """
    user_idx = user_id_to_index[user_id]
    item_idx = movie_id_to_index[item_id]

    # Get user's ratings and filter rated items
    user_ratings = train_sparse_matrix[user_idx, :].toarray().flatten()
    rated_indices = np.where(user_ratings > 0)[0]

    # Get similarities and select top N neighbors
    similarities = similarity_matrix[item_idx, rated_indices]
    top_indices = np.argsort(-similarities)[:N]
    top_similarities = similarities[top_indices]
    top_ratings = user_ratings[rated_indices[top_indices]]

    # Calculate weighted average
    return np.dot(top_ratings, top_similarities) / np.sum(top_similarities) if np.sum(top_similarities) > 0 else np.nan

def predict_weighted_popularity(user_id, item_id, train_sparse_matrix, movie_ids, movie_id_to_index, user_id_to_index, similarity_matrix, item_popularity, favor_popular, N):
    """
    Predict rating using popularity-adjusted similarity weights
    Args:
        user_id: Target user ID
        item_id: Target movie ID
        favor_popular: Boolean flag to favor popular or unpopular items
        Other args: Model parameters and data structures
    Returns:
        Predicted rating or NaN if no valid neighbors
    """
    user_idx = user_id_to_index[user_id]
    item_idx = movie_id_to_index[item_id]

    # Get user's ratings and filter rated items
    user_ratings = train_sparse_matrix[user_idx, :].toarray().flatten()
    rated_indices = np.where(user_ratings > 0)[0]

    # Get similarities and select top N neighbors
    similarities = similarity_matrix[item_idx, rated_indices]
    top_indices = np.argsort(-similarities)[:N]
    top_similarities = similarities[top_indices]
    top_ratings = user_ratings[rated_indices[top_indices]]
    
    # Calculate popularity-adjusted weights
    top_popularities = np.array([item_popularity.get(movie_ids[idx], 0) for idx in rated_indices[top_indices]])
    weights = top_similarities * (1 + np.log1p(top_popularities)) if favor_popular else \
              top_similarities / (1 + np.log1p(top_popularities))

    # Calculate weighted average
    return np.dot(top_ratings, weights) / np.sum(weights) if np.sum(weights) > 0 else np.nan

def run_experiment_1():
    """
    Execute Experiment 1: Compare different neighborhood sizes (N)
    with fixed training ratio (T=80%) and no filtering
    """
    start_time = time.time()
    print("Starting Experiment 1...")
    
    # Prepare data and similarity matrices
    train_ratings, test_ratings = train_test_split_ratings(ratings)
    train_sparse_matrix, train_movie_ids, train_user_ids = create_sparse_matrix(train_ratings)
    item_popularity = train_ratings['movieId'].value_counts().to_dict()

    print("Computing Similarity Matrices...")
    similarity_matrices = {
        'cosine': compute_similarity(train_sparse_matrix, 'cosine'),
        'pearson': compute_similarity(train_sparse_matrix, 'pearson')
    }

    # Test different neighborhood sizes
    N_values = [5, 10, 15, 20, 25]
    results = []
    
    for N in N_values:
        print(f"\n{'='*40}\nProcessing N={N}")
        
        # Compare both similarity metrics and weighting strategies
        for sim_name, similarity_matrix in similarity_matrices.items():
            for favor_popular in [None, True, False]:
                mode = "Weighted Average"
                if favor_popular is True: mode = "Popular Favoring"
                if favor_popular is False: mode = "Unpopular Favoring"
                
                # Generate predictions and calculate metrics
                predicted_ratings = parallel_prediction(
                    test_ratings, train_sparse_matrix, train_movie_ids, 
                    train_user_ids, similarity_matrix, item_popularity, 
                    favor_popular, N
                )
                
                mae, precision, recall = calculate_metrics(test_ratings, predicted_ratings)
                results.append({
                    'N': N,
                    'Similarity': sim_name,
                    'Favor Popular': favor_popular,
                    'MAE': mae,
                    'Precision': precision,
                    'Recall': recall
                })

    # Output results
    total_time = time.time() - start_time
    print(f"\nExperiment 1 completed in {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s")
    print(tabulate(results, headers="keys", tablefmt="fancy_grid"))

def run_experiment_2(best_N):
    """
    Execute Experiment 2: Compare different training ratios (T)
    using the best neighborhood size (N) from Experiment 1
    """
    start_time = time.time()
    print("Starting Experiment 2...")
    
    T_values = [0.5, 0.7, 0.9]
    results = []
    
    for T in T_values:
        # Split data with current training ratio
        train_ratings, test_ratings = train_test_split_ratings(ratings, test_size=(1 - T))
        train_sparse_matrix, train_movie_ids, train_user_ids = create_sparse_matrix(train_ratings)
        item_popularity = train_ratings['movieId'].value_counts().to_dict()

        # Compute similarity matrices
        similarity_matrices = {
            'cosine': compute_similarity(train_sparse_matrix, 'cosine'),
            'pearson': compute_similarity(train_sparse_matrix, 'pearson')
        }

        # Compare both similarity metrics and weighting strategies
        for sim_name, similarity_matrix in similarity_matrices.items():
            for favor_popular in [None, True, False]:
                # Generate predictions and calculate metrics
                predicted_ratings = parallel_prediction(
                    test_ratings, train_sparse_matrix, train_movie_ids, 
                    train_user_ids, similarity_matrix, item_popularity, 
                    favor_popular, best_N
                )
                
                mae, precision, recall = calculate_metrics(test_ratings, predicted_ratings)
                results.append({
                    'T': T,
                    'Similarity': sim_name,
                    'Favor Popular': favor_popular,
                    'MAE': mae,
                    'Precision': precision,
                    'Recall': recall
                })

    # Output results
    total_time = time.time() - start_time
    print(f"\nExperiment 2 completed in {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s")
    print(tabulate(results, headers="keys", tablefmt="fancy_grid"))

if __name__ == "__main__":
    """
    Main execution block: Handle command line arguments
    Usage:
        - Run Experiment 1: python RecommendationSystem.py -1
        - Run Experiment 2: python RecommendationSystem.py -2
    """
    if len(sys.argv) < 2:
        print("Usage: python RecommendationSystem.py -1 (Experiment 1) or -2 (Experiment 2)")
        sys.exit(1)

    if sys.argv[1] == "-1":
        run_experiment_1()
    elif sys.argv[1] == "-2":
        best_N = int(input("Enter best N from Experiment 1: "))
        run_experiment_2(best_N)