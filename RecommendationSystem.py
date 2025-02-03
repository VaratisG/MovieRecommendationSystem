import pandas as pd
import time
import warnings
from tabulate import tabulate
import sys
from utils import (
    train_test_split_ratings,
    create_sparse_matrix,
    compute_similarity,
    parallel_prediction,
    calculate_metrics,
    filter_ratings,
)

warnings.filterwarnings('ignore')

# Load movie ratings dataset from CSV file
ratings = pd.read_csv("csv/ratings.csv")

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
    user_avg_ratings = train_ratings.groupby('userId')['rating'].mean().to_dict()

    print("Computing Similarity Matrices...")
    similarity_matrices = {
        'cosine': compute_similarity(train_sparse_matrix, 'cosine'),
        'pearson': compute_similarity(train_sparse_matrix, 'pearson')
    }

    # Test different neighborhood sizes
    N_values = [5, 20, 50, 100, 150]
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
                
                mae, precision, recall = calculate_metrics(test_ratings, predicted_ratings, user_avg_ratings)
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
        user_avg_ratings = train_ratings.groupby('userId')['rating'].mean().to_dict()

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
                
                mae, precision, recall = calculate_metrics(test_ratings, predicted_ratings, user_avg_ratings)
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


def run_experiment_3(best_N):
    """
    Execute Experiment 3: Test different M/M' filtering combinations
    using best N from Experiment 1 and T=80%.
    """
    start_time = time.time()
    print("Starting Experiment 3...")
    
    # Define filtering combinations (M, M') to test
    filter_values = [(5, 5), (10, 10), (20, 20)]  # Example values, adjust as needed
    results = []
    
    for M, M_prime in filter_values:
        print(f"\n{'='*40}\nProcessing M={M}, M'={M_prime}")
        
        # Apply filtering using your preprocessing function
        filtered_ratings = filter_ratings(M, M_prime)
        
        # Split data with T=80%
        train_ratings, test_ratings = train_test_split_ratings(filtered_ratings, test_size=0.2)
        train_sparse_matrix, train_movie_ids, train_user_ids = create_sparse_matrix(train_ratings)
        item_popularity = train_ratings['movieId'].value_counts().to_dict()

        print("Computing Similarity Matrices...")
        similarity_matrices = {
            'cosine': compute_similarity(train_sparse_matrix, 'cosine'),
            'pearson': compute_similarity(train_sparse_matrix, 'pearson')
        }

        # Test configurations
        for sim_name, similarity_matrix in similarity_matrices.items():
            for favor_popular in [None, True, False]:
                mode = "Weighted Average"
                if favor_popular is True: mode = "Popular Favoring"
                if favor_popular is False: mode = "Unpopular Favoring"
                
                print(f"\n- {sim_name} similarity | {mode}")
                
                # Generate predictions
                predicted_ratings = parallel_prediction(
                    test_ratings, train_sparse_matrix, train_movie_ids,
                    train_user_ids, similarity_matrix, item_popularity,
                    favor_popular, best_N
                )
                
                # Calculate metrics
                mae, precision, recall = calculate_metrics(test_ratings, predicted_ratings)
                
                # Calculate matrix density
                num_users, num_items = train_sparse_matrix.shape
                density = (train_sparse_matrix.nnz / (num_users * num_items)) * 100
                
                # Store results
                results.append({
                    'M': M,
                    'M_prime': M_prime,
                    'Similarity': sim_name,
                    'Favor Popular': favor_popular,
                    'MAE': mae,
                    'Precision': precision,
                    'Recall': recall,
                    'Density (%)': f"{density:.4f}%"
                })

    # Output results
    total_time = time.time() - start_time
    print(f"\nExperiment 3 completed in {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s")
    print(tabulate(results, headers="keys", tablefmt="fancy_grid"))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py -1 (Experiment 1) or -2 (Experiment 2) or -3 (Experiment 3)")
        sys.exit(1)

    if sys.argv[1] == "-1":
        run_experiment_1()
    elif sys.argv[1] == "-2":
        best_N = int(input("Enter best N from Experiment 1: "))
        run_experiment_2(best_N)
    elif sys.argv[1] == "-3":
        best_N = int(input("Enter best N from Experiment 1: "))
        run_experiment_3(best_N)