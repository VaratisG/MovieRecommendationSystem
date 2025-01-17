import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import os
from concurrent.futures import ThreadPoolExecutor
import time

# Load the dataset
ratings = pd.read_csv("filtered_ratings.csv")

# Create the User-Item matrix
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
user_item_matrix_filled = user_item_matrix.fillna(0)

# Helper function to split data into training and testing
def train_test_split_ratings(ratings, test_size=0.2, random_state=42):
    train, test = train_test_split(ratings, test_size=test_size, random_state=random_state)
    return train, test

# Split ratings into training and testing
train_ratings, test_ratings = train_test_split_ratings(ratings, test_size=0.2)

# Reconstruct the User-Item matrices for training and testing
train_matrix = train_ratings.pivot(index='userId', columns='movieId', values='rating')
test_matrix = test_ratings.pivot(index='userId', columns='movieId', values='rating')
train_matrix_filled = train_matrix.fillna(0)

# Compute Cosine similarity
print("Computing Cosine Similarity... (This might take some time)")
start_time = time.time()
cosine_sim = cosine_similarity(train_matrix_filled.T)
cosine_sim_df = pd.DataFrame(cosine_sim, index=train_matrix.columns, columns=train_matrix.columns)
print(f"Cosine Similarity computed in {time.time() - start_time:.2f} seconds.")

# Compute Pearson similarity
print("Computing Pearson Correlation... (This might take some time)")
start_time = time.time()
pearson_sim = train_matrix.corr(method='pearson')
print(f"Pearson Correlation computed in {time.time() - start_time:.2f} seconds.")

# Compute item popularity
item_popularity = train_ratings['movieId'].value_counts()

# Prediction functions
def predict_weighted_average(user_id, item_id, user_item_matrix, similarity_matrix, N=5):
    if item_id not in similarity_matrix.columns or user_id not in user_item_matrix.index:
        return np.nan
    similar_items = similarity_matrix[item_id].drop(item_id).sort_values(ascending=False)
    user_rated_items = user_item_matrix.loc[user_id].dropna()
    rated_and_similar = similar_items[similar_items.index.isin(user_rated_items.index)].head(N)
    if rated_and_similar.empty:
        return np.nan
    ratings = user_rated_items[rated_and_similar.index]
    weights = rated_and_similar
    prediction = np.dot(ratings, weights) / weights.sum()
    return prediction

def predict_with_popularity(user_id, item_id, user_item_matrix, similarity_matrix, item_popularity, favor_popular=True, N=5):
    prediction = predict_weighted_average(user_id, item_id, user_item_matrix, similarity_matrix, N)
    if np.isnan(prediction):
        return np.nan
    popularity_score = item_popularity[item_id] if item_id in item_popularity else 0
    return prediction + (0.1 * popularity_score if favor_popular else -0.1 * popularity_score)

# Evaluation metrics
def calculate_metrics(test_matrix, predicted_ratings):
    mae_values = []
    user_means = test_matrix.mean(axis=1)
    relevant_items = {u: set(test_matrix.loc[u][test_matrix.loc[u] >= user_means[u]].index) for u in test_matrix.index}
    predicted_relevant = {}
    for user, user_preds in predicted_ratings.items():
        user_relevant_items = {item for item, pred in user_preds.items() if pred >= user_means[user]}
        predicted_relevant[user] = user_relevant_items
        for item, pred in user_preds.items():
            if not np.isnan(pred) and not np.isnan(test_matrix.loc[user, item]):
                mae_values.append(abs(test_matrix.loc[user, item] - pred))
    mae = np.mean(mae_values)
    precision = np.mean([
        len(predicted_relevant[u] & relevant_items[u]) / len(predicted_relevant[u])
        if len(predicted_relevant[u]) > 0 else 0
        for u in test_matrix.index
    ])
    recall = np.mean([
        len(predicted_relevant[u] & relevant_items[u]) / len(relevant_items[u])
        if len(relevant_items[u]) > 0 else 0
        for u in test_matrix.index
    ])
    return mae, precision, recall

# Parallel prediction
def parallel_predictions(test_matrix, similarity_matrix, item_popularity, favor_popular, N):
    predicted_ratings = {}
    def process_user(user):
        user_preds = {}
        for item in test_matrix.columns:
            if favor_popular is None:
                user_preds[item] = predict_weighted_average(user, item, train_matrix_filled, similarity_matrix, N)
            else:
                user_preds[item] = predict_with_popularity(user, item, train_matrix_filled, similarity_matrix, item_popularity, favor_popular, N)
        return user, user_preds
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_user, test_matrix.index)
    for user, user_preds in results:
        predicted_ratings[user] = user_preds
    return predicted_ratings

# Experiment setup
N_values = [5, 10, 15, 20, 25]
results = []

print("Starting predictions in parallel... This may run in the background.")
start_time = time.time()
for N in N_values:
    for sim_name, similarity_matrix in zip(['cosine', 'pearson'], [cosine_sim_df, pearson_sim]):
        for favor_popular in [None, True, False]:
            predicted_ratings = parallel_predictions(test_matrix, similarity_matrix, item_popularity, favor_popular, N)
            mae, precision, recall = calculate_metrics(test_matrix, predicted_ratings)
            results.append({
                'N': N,
                'Similarity': sim_name,
                'Favor Popular': favor_popular,
                'MAE': mae,
                'Precision': precision,
                'Recall': recall
            })

print(f"Predictions completed in {time.time() - start_time:.2f} seconds.")

# Convert results to DataFrame and display
results_df = pd.DataFrame(results)
print(results_df)

# Save results
results_df.to_csv('csv/experiment_results.csv', index=False)
print("Results saved to 'csv/experiment_results.csv'")
