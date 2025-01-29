import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from multiprocessing import Pool, shared_memory
import time
import gc
import warnings
from tabulate import tabulate
import sys
import os

warnings.filterwarnings('ignore')

# Load the dataset
ratings = pd.read_csv("csv/ratings.csv")

#Print and progress and count the running time
def print_progress(current, total, start_time):
    elapsed = time.time() - start_time
    percent = (current / total) * 100
    remaining = (elapsed / (current + 1)) * (total - current) if current > 0 else 0
    print(f"\rProgress: {percent:.1f}% | Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s", end="")

# Split ratings into training and testing
def train_test_split_ratings(ratings, test_size=0.2, random_state=42):
    train, test = train_test_split(ratings, test_size=test_size, random_state=random_state)
    return train, test

# Create sparse matrix
def create_sparse_matrix(ratings_df):
    user_item_matrix = ratings_df.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)
    sparse_matrix = csr_matrix(user_item_matrix.values)
    return sparse_matrix, list(user_item_matrix.columns), list(user_item_matrix.index)

# Compute Cosine similarity and Pearson (optimized for sparse matrices)
def compute_similarity(sparse_matrix, method='cosine'):
    if method == 'cosine':
        return cosine_similarity(sparse_matrix.T, dense_output=True)  # Return dense array
    elif method == 'pearson':
        # Sparse Pearson computation
        user_means = sparse_matrix.mean(axis=1).A1
        centered_matrix = sparse_matrix.copy()
        centered_matrix.data -= user_means[sparse_matrix.nonzero()[0]]
        
        # Compute Pearson using sparse covariance
        cov = centered_matrix.T @ centered_matrix
        std_dev = np.sqrt(np.array(cov.diagonal()).clip(min=1e-9))
        pearson = cov / np.outer(std_dev, std_dev)
        return pearson.toarray()  # Return dense array
    else:
        raise ValueError("Invalid similarity method. Choose 'cosine' or 'pearson'.")

# Prediction helper function
def predict(row, train_sparse_matrix, train_movie_ids, train_user_ids, similarity_matrix, item_popularity, favor_popular, N):
    user_id = row['userId']
    item_id = row['movieId']
    if favor_popular is None:
        return predict_weighted_average(user_id, item_id, train_sparse_matrix, train_movie_ids, train_user_ids, similarity_matrix, N)
    else:
        return predict_weighted_popularity(user_id, item_id, train_sparse_matrix, train_movie_ids, train_user_ids, similarity_matrix, item_popularity, favor_popular, N)

# Parallel prediction with shared memory
def parallel_prediction(test_ratings, train_sparse_matrix, train_movie_ids, train_user_ids, similarity_matrix, item_popularity, favor_popular, N):
    total_items = len(test_ratings)
    start_time = time.time()
    
    if isinstance(similarity_matrix, csr_matrix):
        similarity_matrix = similarity_matrix.toarray()

    shm = shared_memory.SharedMemory(create=True, size=similarity_matrix.nbytes)
    shared_sim = np.ndarray(similarity_matrix.shape, dtype=similarity_matrix.dtype, buffer=shm.buf)
    np.copyto(shared_sim, similarity_matrix)

    print(f"\nStarting predictions for N={N}...")
    
    with Pool(processes=4) as pool:
        results = []
        for i, res in enumerate(pool.starmap(predict, 
            [(row, train_sparse_matrix, train_movie_ids, train_user_ids, shared_sim, 
              item_popularity, favor_popular, N) 
             for _, row in test_ratings.iterrows()])):
            results.append(res)
            if i % 100 == 0:
                print_progress(i, total_items, start_time)
    
    shm.close()
    shm.unlink()
    
    print_progress(total_items, total_items, start_time)
    print(f"\nPredictions completed for N={N} in {time.time()-start_time:.1f}s")
    return results

# Calculate MAE, precision, and recall
def calculate_metrics(test_ratings, predicted_ratings, train_user_ids, train_movie_ids):
    test_ratings = test_ratings.reset_index(drop=True)
    mae = np.nanmean(np.abs(test_ratings['rating'] - predicted_ratings))

    true_positives = 0
    false_positives = 0
    false_negatives = 0

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

# Prediction functions
def predict_weighted_average(user_id, item_id, train_sparse_matrix, movie_ids, user_ids, similarity_matrix, N):
    if item_id not in movie_ids or user_id not in user_ids:
        return np.nan

    user_index = user_ids.index(user_id)
    item_index = movie_ids.index(item_id)

    user_ratings = train_sparse_matrix[user_index, :].toarray().flatten()
    rated_indices = np.where(user_ratings > 0)[0]

    # Use dense similarity matrix
    similarities = similarity_matrix[item_index, rated_indices]
    top_indices = np.argsort(-similarities)[:N]
    top_similarities = similarities[top_indices]
    top_ratings = user_ratings[rated_indices[top_indices]]

    if np.sum(top_similarities) == 0:
        return np.nan

    return np.dot(top_ratings, top_similarities) / np.sum(top_similarities)

def predict_weighted_popularity(user_id, item_id, train_sparse_matrix, movie_ids, user_ids, similarity_matrix, item_popularity, favor_popular, N):
    if item_id not in movie_ids or user_id not in user_ids:
        return np.nan

    user_index = user_ids.index(user_id)
    item_index = movie_ids.index(item_id)

    user_ratings = train_sparse_matrix[user_index, :].toarray().flatten()
    rated_indices = np.where(user_ratings > 0)[0]

    # Use dense similarity matrix
    similarities = similarity_matrix[item_index, rated_indices]
    top_indices = np.argsort(-similarities)[:N]
    top_similarities = similarities[top_indices]
    top_ratings = user_ratings[rated_indices[top_indices]]
    top_popularities = np.array([item_popularity.get(movie_ids[idx], 0) for idx in rated_indices[top_indices]])

    weights = top_similarities * (1 + np.log1p(top_popularities)) if favor_popular else \
              top_similarities / (1 + np.log1p(top_popularities))

    if np.sum(weights) == 0:
        return np.nan

    return np.dot(top_ratings, weights) / np.sum(weights)

# Experiment 1: Χωρίς φιλτράρισμα, για Τ=80%, και για 5 τιμές Ν
def run_experiment_1():
    start_time = time.time()
    print("Starting Experiment 1...")
    
    train_ratings, test_ratings = train_test_split_ratings(ratings, test_size=0.2)
    train_sparse_matrix, train_movie_ids, train_user_ids = create_sparse_matrix(train_ratings)
    item_popularity = train_ratings['movieId'].value_counts()

    print("Computing Similarity Matrices...")
    sim_start = time.time()
    similarity_matrices = {
        'cosine': compute_similarity(train_sparse_matrix, 'cosine'),
        'pearson': compute_similarity(train_sparse_matrix, 'pearson')
    }
    print(f"Similarity matrices computed in {time.time()-sim_start:.1f}s")

    N_values = [5, 10, 15, 20, 25]
    results = []
    
    for N in N_values:
        exp_start = time.time()
        print(f"\n{'='*40}\nProcessing N={N}")
        
        for sim_name, similarity_matrix in similarity_matrices.items():
            print(f"\nProcessing {sim_name} similarity...")
            
            for favor_popular in [None, True, False]:
                mode = "Weighted Average"
                if favor_popular is True: mode = "Popular Favoring"
                if favor_popular is False: mode = "Unpopular Favoring"
                print(f"\n- {mode}")
                
                pred_start = time.time()
                predicted_ratings = parallel_prediction(
                    test_ratings, train_sparse_matrix, train_movie_ids, 
                    train_user_ids, similarity_matrix, item_popularity, 
                    favor_popular, N
                )
                
                mae, precision, recall = calculate_metrics(test_ratings, predicted_ratings, 
                                                          train_user_ids, train_movie_ids)
                results.append({
                    'N': N,
                    'Similarity': sim_name,
                    'Favor Popular': favor_popular,
                    'MAE': mae,
                    'Precision': precision,
                    'Recall': recall
                })
                print(f"Completed in {time.time()-pred_start:.1f}s")

    total_time = time.time() - start_time
    print(f"\n{'='*40}\nExperiment 1 completed in {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s")
    print(tabulate(results, headers="keys", tablefmt="fancy_grid"))

# Experiment 2: Χωρίς φιλτράρισμα, και με το καλύτερο Ν
def run_experiment_2(best_N):
    start_time = time.time()
    print("Starting Experiment 2...")
    
    T_values = [0.5, 0.7, 0.9]
    results = []
    
    for T in T_values:
        exp_start = time.time()
        print(f"\n{'='*40}\nProcessing T={T*100}% split")
        
        train_ratings, test_ratings = train_test_split_ratings(ratings, test_size=(1 - T))
        train_sparse_matrix, train_movie_ids, train_user_ids = create_sparse_matrix(train_ratings)
        item_popularity = train_ratings['movieId'].value_counts()

        similarity_matrices = {
            'cosine': compute_similarity(train_sparse_matrix, 'cosine'),
            'pearson': compute_similarity(train_sparse_matrix, 'pearson')
        }

        for sim_name, similarity_matrix in similarity_matrices.items():
            print(f"\nProcessing {sim_name} similarity...")
            
            for favor_popular in [None, True, False]:
                mode = "Weighted Average"
                if favor_popular is True: mode = "Popular Favoring"
                if favor_popular is False: mode = "Unpopular Favoring"
                print(f"\n- {mode}")
                
                pred_start = time.time()
                predicted_ratings = parallel_prediction(
                    test_ratings, train_sparse_matrix, train_movie_ids, 
                    train_user_ids, similarity_matrix, item_popularity, 
                    favor_popular, best_N
                )
                
                mae, precision, recall = calculate_metrics(test_ratings, predicted_ratings, 
                                                          train_user_ids, train_movie_ids)
                results.append({
                    'T': T,
                    'Similarity': sim_name,
                    'Favor Popular': favor_popular,
                    'MAE': mae,
                    'Precision': precision,
                    'Recall': recall
                })
                print(f"Completed in {time.time()-pred_start:.1f}s")

    total_time = time.time() - start_time
    print(f"\n{'='*40}\nExperiment 2 completed in {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s")
    print(tabulate(results, headers="keys", tablefmt="fancy_grid"))

#Main Script
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py -1 (Experiment 1) or -2 (Experiment 2)")
        sys.exit(1)

    start_time = time.time()
    
    if sys.argv[1] == "-1":
        run_experiment_1()
    elif sys.argv[1] == "-2":
        best_N = int(input("Enter best N from Experiment 1: "))
        run_experiment_2(best_N)
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s")