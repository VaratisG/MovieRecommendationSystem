import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from multiprocessing import Pool
import time
import gc
import warnings
from tabulate import tabulate

warnings.filterwarnings('ignore')


# Load the dataset
ratings = pd.read_csv("csv/ratings.csv")


# Split ratings into training and testing
def train_test_split_ratings(ratings, test_size=0.2, random_state=42):
    train, test = train_test_split(ratings, test_size=test_size, random_state=random_state)
    return train, test


# Create sparse matrix
def create_sparse_matrix(ratings_df):
    """Creates a sparse matrix from the ratings dataframe."""
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
            pearson_corr[std_dev == 0, :] = 0
            pearson_corr[:, std_dev == 0] = 0

        return csr_matrix(np.nan_to_num(pearson_corr))
    else:
        raise ValueError("Invalid similarity method. Choose 'cosine' or 'pearson'.")


# Prediction functions
def predict_weighted_average(user_id, item_id, train_sparse_matrix, movie_ids, user_ids, similarity_matrix, N=5):
    if item_id not in movie_ids or user_id not in user_ids:
        return np.nan

    user_index = user_ids.index(user_id)
    item_index = movie_ids.index(item_id)

    similar_items = similarity_matrix[item_index].toarray().flatten()
    similar_items[item_index] = 0
    top_similar_indices = np.argsort(similar_items)[::-1][:N]

    user_ratings = train_sparse_matrix[user_index].toarray().flatten()
    relevant_ratings = user_ratings[top_similar_indices]
    relevant_similarities = similar_items[top_similar_indices]

    if np.sum(relevant_similarities) == 0:
        return np.nan

    return np.dot(relevant_ratings, relevant_similarities) / np.sum(relevant_similarities)


def predict_with_popularity(user_id, item_id, train_sparse_matrix, movie_ids, user_ids, similarity_matrix, item_popularity, favor_popular=True, N=5):
    prediction = predict_weighted_average(user_id, item_id, train_sparse_matrix, movie_ids, user_ids, similarity_matrix, N)
    if np.isnan(prediction):
        return np.nan

    popularity_score = item_popularity[item_id] if item_id in item_popularity else 0
    return prediction + (0.1 * popularity_score if favor_popular else -0.1 * popularity_score)


# Parallel prediction
def predict_for_user(user_id, test_ratings_user, train_sparse_matrix, movie_ids, user_ids, similarity_matrix, item_popularity, favor_popular, N):
    user_preds = {}
    if user_id not in user_ids:
        return user_id, user_preds

    for item_id in test_ratings_user['movieId'].tolist():
        if favor_popular is None:
            user_preds[item_id] = predict_weighted_average(user_id, item_id, train_sparse_matrix, movie_ids, user_ids, similarity_matrix, N)
        elif favor_popular is True:
            user_preds[item_id] = predict_with_popularity(user_id, item_id, train_sparse_matrix, movie_ids, user_ids, similarity_matrix, item_popularity, favor_popular, N)
        elif favor_popular is False:
            user_preds[item_id] = predict_with_popularity(user_id, item_id, train_sparse_matrix, movie_ids, user_ids, similarity_matrix, item_popularity, False, N)
    return user_id, user_preds


def parallel_prediction(test_ratings, train_sparse_matrix, movie_ids, user_ids, similarity_matrix, item_popularity, favor_popular, N, processes=4):
    all_user_ids = test_ratings['userId'].unique()
    total_users = len(all_user_ids)
    results = {}

    with Pool(processes=processes) as pool:
        args_list = [(user_id, test_ratings[test_ratings['userId'] == user_id], train_sparse_matrix, movie_ids, user_ids, similarity_matrix, item_popularity, favor_popular, N)
                     for user_id in all_user_ids]

        for i, (user_id, user_preds) in enumerate(pool.starmap(predict_for_user, args_list)):
            results[user_id] = user_preds
            progress = (i + 1) / total_users * 100
            print(f"Prediction Progress: {progress:.2f}% complete", end='\r')
    print()
    return results


# Evaluation metrics
def calculate_metrics(test_ratings, predicted_ratings, user_ids, movie_ids):
    mae_values = []
    precision_values = []
    recall_values = []

    for user_id in test_ratings['userId'].unique():
        if user_id not in user_ids:
            continue
        test_user_ratings = test_ratings[test_ratings['userId'] == user_id]
        user_preds = predicted_ratings.get(user_id, {})

        user_actual_ratings = test_user_ratings.set_index('movieId')['rating'].to_dict()
        user_mean_rating = test_user_ratings['rating'].mean()

        valid_predictions = {item_id: pred for item_id, pred in user_preds.items() if item_id in user_actual_ratings and not np.isnan(pred)}

        for item_id, pred in valid_predictions.items():
            mae_values.append(abs(user_actual_ratings[item_id] - pred))

        relevant_items = set(user_actual_ratings.keys())
        predicted_relevant = set(item_id for item_id, pred in valid_predictions.items() if pred >= user_mean_rating)

        precision_values.append(len(predicted_relevant & relevant_items) / len(predicted_relevant) if predicted_relevant else 0)
        recall_values.append(len(predicted_relevant & relevant_items) / len(relevant_items) if relevant_items else 0)

    mae = np.mean(mae_values) if mae_values else np.nan
    precision = np.mean(precision_values) if precision_values else np.nan
    recall = np.mean(recall_values) if recall_values else np.nan

    return mae, precision, recall


# Main experiment function
def run_experiment():
    results = []
    N_values = [5, 10, 15, 20, 25]

    similarity_matrices = {
        'cosine': compute_similarity(train_sparse_matrix, method='cosine'),
        'pearson': compute_similarity(train_sparse_matrix, method='pearson')
    }

    for N in N_values:
        for sim_name, similarity_matrix in similarity_matrices.items():
            for favor_popular in [None, True, False]:
                start_time = time.time()
                predicted_ratings = parallel_prediction(
                    test_ratings, train_sparse_matrix, train_movie_ids, train_user_ids, similarity_matrix, item_popularity, favor_popular, N, processes=4
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
                print(f"Completed: N={N}, Sim={sim_name}, Favor Popular={favor_popular}, Time: {time.time() - start_time:.2f}s")
                gc.collect()

    return pd.DataFrame(results)


# Run the script
if __name__ == "__main__":
    print("Experiment started...")
    start_time = time.time()

    train_ratings, test_ratings = train_test_split_ratings(ratings, test_size=0.2)
    train_sparse_matrix, train_movie_ids, train_user_ids = create_sparse_matrix(train_ratings)
    test_sparse_matrix, test_movie_ids, test_user_ids = create_sparse_matrix(test_ratings)
    item_popularity = train_ratings['movieId'].value_counts()

    experiment_results = run_experiment()

    print("\nFinal Results:\n")
    print(tabulate(experiment_results, headers="keys", tablefmt="fancy_grid"))

    print(f"\nExperiment completed in {time.time() - start_time:.2f} seconds.")
