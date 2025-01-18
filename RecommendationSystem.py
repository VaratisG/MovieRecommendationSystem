import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from multiprocessing import Pool
import time

# Load the dataset
ratings = pd.read_csv("csv/ratings.csv")

# Create the User-Item sparse matrix
def create_sparse_matrix(ratings):
    user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
    sparse_matrix = csr_matrix(user_item_matrix.fillna(0).values)
    return sparse_matrix, list(user_item_matrix.columns), list(user_item_matrix.index)

sparse_matrix, movie_ids, user_ids = create_sparse_matrix(ratings)

# Helper function to split data into training and testing
def train_test_split_ratings(ratings, test_size=0.2, random_state=42):
    train, test = train_test_split(ratings, test_size=test_size, random_state=random_state)
    return train, test

# Split ratings into training and testing
train_ratings, test_ratings = train_test_split_ratings(ratings, test_size=0.2)

# Create sparse matrices for training and testing
train_sparse_matrix, train_movie_ids, train_user_ids = create_sparse_matrix(train_ratings)
test_sparse_matrix, test_movie_ids, test_user_ids = create_sparse_matrix(test_ratings)

# Compute Cosine similarity
cosine_sim = cosine_similarity(train_sparse_matrix.T, dense_output=False)

# Compute item popularity
item_popularity = train_ratings['movieId'].value_counts()

# Prediction functions
# Adjusted prediction and metrics functions
def predict_weighted_average(user_id, item_id, user_item_matrix, similarity_matrix, N=5):
    if item_id not in movie_ids or user_id not in user_ids:
        return np.nan

    user_index = user_ids.index(user_id)
    item_index = movie_ids.index(item_id)

    # Find similar items
    similar_items = similarity_matrix[item_index].toarray().flatten()
    similar_items[item_index] = 0  # Exclude the item itself
    top_similar_indices = np.argsort(similar_items)[::-1][:N]

    # Extract ratings for similar items
    user_ratings = user_item_matrix[user_index].toarray().flatten()
    relevant_ratings = user_ratings[top_similar_indices]
    relevant_similarities = similar_items[top_similar_indices]

    if np.sum(relevant_similarities) == 0:
        return np.nan

    # Weighted average
    prediction = np.dot(relevant_ratings, relevant_similarities) / np.sum(relevant_similarities)
    return prediction


def predict_with_popularity(user_id, item_id, user_item_matrix, similarity_matrix, item_popularity, favor_popular=True, N=5):
    prediction = predict_weighted_average(user_id, item_id, user_item_matrix, similarity_matrix, N)
    if np.isnan(prediction):
        return np.nan

    popularity_score = item_popularity[item_id] if item_id in item_popularity else 0
    return prediction + (0.1 * popularity_score if favor_popular else -0.1 * popularity_score)


# Parallel prediction
def predict_for_user(user_id, test_matrix, train_sparse_matrix, similarity_matrix, item_popularity, favor_popular, N):
    user_preds = {}

    if user_id not in user_ids:
        return user_id, user_preds

    for item_id in movie_ids:
        if favor_popular is None:
            user_preds[item_id] = predict_weighted_average(user_id, item_id, train_sparse_matrix, similarity_matrix, N)
        else:
            user_preds[item_id] = predict_with_popularity(user_id, item_id, train_sparse_matrix, similarity_matrix, item_popularity, favor_popular, N)

    return user_id, user_preds


def parallel_prediction(test_matrix, train_sparse_matrix, similarity_matrix, item_popularity, favor_popular, N, processes=4):
    with Pool(processes=processes) as pool:
        results = pool.starmap(
            predict_for_user,
            [(user_id, test_matrix, train_sparse_matrix, similarity_matrix, item_popularity, favor_popular, N)
             for user_id in test_user_ids]
        )
    return {user_id: preds for user_id, preds in results}


# Evaluation metrics
def calculate_metrics(test_matrix, predicted_ratings):
    mae_values = []
    user_means = np.array(test_matrix.mean(axis=1)).flatten()  # Compute user means for binary relevance

    # Create dictionaries of relevant items for users
    relevant_items = {user_ids[i]: set(test_matrix[i].indices) for i in range(len(user_ids))}

    predicted_relevant = {}
    for user_id, user_preds in predicted_ratings.items():
        if user_id not in user_ids:
            continue

        # Predicted relevant items for the user
        user_relevant_items = {item_id for item_id, pred in user_preds.items() if pred >= user_means[user_ids.index(user_id)]}
        predicted_relevant[user_id] = user_relevant_items

        # Calculate MAE for each user
        for item_id, pred in user_preds.items():
            if item_id not in movie_ids or user_id not in user_ids:
                continue

            user_idx = user_ids.index(user_id)
            item_idx = movie_ids.index(item_id)

            # Safely access the test matrix
            if item_idx < test_matrix.shape[1] and user_idx < test_matrix.shape[0]:
                actual = test_matrix[user_idx, item_idx]
                if actual != 0 and not np.isnan(pred):
                    mae_values.append(abs(actual - pred))

    # Calculate precision and recall
    precision = np.mean([
        len(predicted_relevant[u] & relevant_items[u]) / len(predicted_relevant[u])
        if len(predicted_relevant[u]) > 0 else 0
        for u in user_ids
    ])

    recall = np.mean([
        len(predicted_relevant[u] & relevant_items[u]) / len(relevant_items[u])
        if len(relevant_items[u]) > 0 else 0
        for u in user_ids
    ])

    mae = np.mean(mae_values)
    return mae, precision, recall

# Main experiment function
def run_experiment():
    results = []
    N_values = [5, 10, 15, 20, 25]

    for N in N_values:
        for sim_name, similarity_matrix in zip(['cosine'], [cosine_sim]):
            for favor_popular in [None, True, False]:
                predicted_ratings = parallel_prediction(
                    test_sparse_matrix, train_sparse_matrix, similarity_matrix, item_popularity, favor_popular, N, processes=4
                )

                mae, precision, recall = calculate_metrics(test_sparse_matrix, predicted_ratings)

                results.append({
                    'N': N,
                    'Similarity': sim_name,
                    'Favor Popular': favor_popular,
                    'MAE': mae,
                    'Precision': precision,
                    'Recall': recall
                })

    return pd.DataFrame(results)


# Run the script
if __name__ == "__main__":
    print("The experiment process has started. Please wait...")

    start_time = time.time()

    # Run the experiment
    experiment_results = run_experiment()

    end_time = time.time()

    # Print results to terminal
    print("\nExperiment Results:")
    print(experiment_results)

    # Save results to a CSV
    experiment_results.to_csv("results.csv", index=False)

    # Display total time taken
    total_time = end_time - start_time
    print(f"\nThe experiment process has completed in {total_time:.2f} seconds.")