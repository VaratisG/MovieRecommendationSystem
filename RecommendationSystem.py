import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
from multiprocessing import Pool
import time
import gc
import warnings
from tabulate import tabulate
import sys

warnings.filterwarnings('ignore')


# Load the dataset
ratings = pd.read_csv("filtered_ratings.csv")


# Split ratings into training and testing
def train_test_split_ratings(ratings, test_size=0.2, random_state=42):
    train, test = train_test_split(ratings, test_size=test_size, random_state=random_state)
    return train, test


# Create sparse matrix
def create_sparse_matrix(ratings_df):
    user_item_matrix = ratings_df.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)
    sparse_matrix = csr_matrix(user_item_matrix.values)
    return sparse_matrix, list(user_item_matrix.columns), list(user_item_matrix.index)


# Compute Cosine similarity and Pearson
def compute_similarity(sparse_matrix, method='cosine'):
    if method == 'cosine':
        return cosine_similarity(sparse_matrix.T, dense_output=False)
    elif method == 'pearson':
        user_means = sparse_matrix.mean(axis=1).A1
        dense_matrix = sparse_matrix.toarray()
        centered_matrix = dense_matrix - user_means[:, None]
        centered_matrix = np.nan_to_num(centered_matrix)

        covariance = np.cov(centered_matrix.T)
        std_dev = np.sqrt(np.diag(covariance))

        with np.errstate(divide='ignore', invalid='ignore'):
            pearson_corr = covariance / np.outer(std_dev, std_dev)
            pearson_corr[np.isnan(pearson_corr)] = 0

        return csr_matrix(pearson_corr)
    else:
        raise ValueError("Invalid similarity method. Choose 'cosine' or 'pearson'.")


# Prediction helper function
def predict(row, train_sparse_matrix, train_movie_ids, train_user_ids, similarity_matrix, item_popularity, favor_popular, N):
    user_id = row['userId']
    item_id = row['movieId']
    if favor_popular is None:
        return predict_weighted_average(user_id, item_id, train_sparse_matrix, train_movie_ids, train_user_ids, N)
    else:
        return predict_weighted_popularity(user_id, item_id, train_sparse_matrix, train_movie_ids, train_user_ids, item_popularity, favor_popular, N)


# Parallel prediction
def parallel_prediction(test_ratings, train_sparse_matrix, train_movie_ids, train_user_ids, similarity_matrix, item_popularity, favor_popular, N):
    with Pool() as pool:
        predictions = pool.starmap(
            predict,
            [(row, train_sparse_matrix, train_movie_ids, train_user_ids, similarity_matrix, item_popularity, favor_popular, N)
             for _, row in test_ratings.iterrows()]
        )
    return predictions


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
def predict_weighted_average(user_id, item_id, train_sparse_matrix, movie_ids, user_ids, N):
    if item_id not in movie_ids or user_id not in user_ids:
        return np.nan

    user_index = user_ids.index(user_id)
    item_index = movie_ids.index(item_id)

    user_ratings = train_sparse_matrix[user_index, :].toarray().flatten()
    rated_indices = np.where(user_ratings > 0)[0]

    similarities = np.array([
        compute_similarity_on_demand(item_index, rated_index, train_sparse_matrix)
        for rated_index in rated_indices
    ])

    top_indices = np.argsort(-similarities)[:N]
    top_similarities = similarities[top_indices]
    top_ratings = user_ratings[rated_indices[top_indices]]

    if np.sum(top_similarities) == 0:
        return np.nan

    return np.dot(top_ratings, top_similarities) / np.sum(top_similarities)


def predict_weighted_popularity(user_id, item_id, train_sparse_matrix, movie_ids, user_ids, item_popularity, favor_popular, N):
    if item_id not in movie_ids or user_id not in user_ids:
        return np.nan

    user_index = user_ids.index(user_id)
    item_index = movie_ids.index(item_id)

    user_ratings = train_sparse_matrix[user_index, :].toarray().flatten()
    rated_indices = np.where(user_ratings > 0)[0]

    similarities = np.array([
        compute_similarity_on_demand(item_index, rated_index, train_sparse_matrix)
        for rated_index in rated_indices
    ])

    top_indices = np.argsort(-similarities)[:N]
    top_similarities = similarities[top_indices]
    top_ratings = user_ratings[rated_indices[top_indices]]
    top_popularities = np.array([item_popularity.get(movie_ids[idx], 0) for idx in rated_indices[top_indices]])

    weights = top_similarities * (1 + np.log1p(top_popularities)) if favor_popular else \
              top_similarities / (1 + np.log1p(top_popularities))

    if np.sum(weights) == 0:
        return np.nan

    return np.dot(top_ratings, weights) / np.sum(weights)


def compute_similarity_on_demand(item1_index, item2_index, train_sparse_matrix):
    item1_vector = train_sparse_matrix[:, item1_index].toarray().flatten()
    item2_vector = train_sparse_matrix[:, item2_index].toarray().flatten()
    if np.all(item1_vector == 0) or np.all(item2_vector == 0):
        return 0
    return 1 - cosine(item1_vector, item2_vector)

# Experiment 1: Χωρίς φιλτράρισμα, για Τ=80%, και για 5 τιμές Ν
def run_experiment_1():
    train_ratings, test_ratings = train_test_split_ratings(ratings, test_size=0.2)
    train_sparse_matrix, train_movie_ids, train_user_ids = create_sparse_matrix(train_ratings)
    test_sparse_matrix, test_movie_ids, test_user_ids = create_sparse_matrix(test_ratings)
    item_popularity = train_ratings['movieId'].value_counts()

    print("Computing Similarity Matrices...")
    N_values = [30, 40, 50, 70, 100]
    similarity_matrices = {
        'cosine': compute_similarity(train_sparse_matrix, method='cosine'),
        'pearson': compute_similarity(train_sparse_matrix, method='pearson')
    }
    
    print("Running Experiment 1...")
    results = []
    for N in N_values:
        for sim_name, similarity_matrix in similarity_matrices.items():
            for favor_popular in [None, True, False]:
                predicted_ratings = parallel_prediction(
                    test_ratings, train_sparse_matrix, train_movie_ids, train_user_ids, similarity_matrix,
                    item_popularity, favor_popular, N
                )
                mae, precision, recall = calculate_metrics(test_ratings, predicted_ratings, train_user_ids, train_movie_ids)
                results.append({
                    'N': N,
                    'Similarity': sim_name,
                    'Favor Popular': favor_popular,
                    'MAE': mae,
                    'Precision': precision,
                    'Recall': recall
                })

    print("\nExperiment 1 Results:\n")
    print(tabulate(results, headers="keys", tablefmt="fancy_grid"))


# Experiment 2: Χωρίς φιλτράρισμα, και με το καλύτερο Ν
def run_experiment_2(best_N):
    T_values = [0.5, 0.7, 0.9]

    results = []
    for T in T_values:
        train_ratings, test_ratings = train_test_split_ratings(ratings, test_size=(1 - T))
        train_sparse_matrix, train_movie_ids, train_user_ids = create_sparse_matrix(train_ratings)
        test_sparse_matrix, test_movie_ids, test_user_ids = create_sparse_matrix(test_ratings)
        item_popularity = train_ratings['movieId'].value_counts()
        
        print("Computing Similarity Matrices...")
        similarity_matrices = {
            'cosine': compute_similarity(train_sparse_matrix, method='cosine'),
            'pearson': compute_similarity(train_sparse_matrix, method='pearson')
        }

        print("Running Experiment 2...")
        for sim_name, similarity_matrix in similarity_matrices.items():
            for favor_popular in [None, True, False]:
                predicted_ratings = parallel_prediction(
                    test_ratings, train_sparse_matrix, train_movie_ids, train_user_ids, similarity_matrix,
                    item_popularity, favor_popular, best_N
                )
                mae, precision, recall = calculate_metrics(test_ratings, predicted_ratings, train_user_ids, train_movie_ids)
                results.append({
                    'T': T,
                    'Similarity': sim_name,
                    'Favor Popular': favor_popular,
                    'MAE': mae,
                    'Precision': precision,
                    'Recall': recall
                })

    print("\nExperiment 2 Results:\n")
    print(tabulate(results, headers="keys", tablefmt="fancy_grid"))


# Main script
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify which experiment to run: -1 for Experiment 1 or -2 for Experiment 2.")
        sys.exit(1)

    experiment = sys.argv[1]
    if experiment == "-1":
        run_experiment_1()
    elif experiment == "-2":
        best_N = int(input("Enter the best N value from Experiment 1: "))
        run_experiment_2(best_N)
    else:
        print("Invalid parameter. Use -1 for Experiment 1 or -2 for Experiment 2.")