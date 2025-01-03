import numpy as np
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def predict_weighted_average(user_id, item_id, user_item_matrix, similarity_matrix, N=10):
    """
    Predict the rating for a given user and item using the weighted average of the N nearest neighbors.
    Similarity is used as the weight.
    """
    # Get the similarity scores for the target item
    item_similarities = similarity_matrix[item_id]
    
    # Get the user's ratings
    user_ratings = user_item_matrix.loc[user_id]
    
    # Combine similarities and ratings for non-zero ratings
    rated_items = user_ratings[user_ratings > 0].index
    similarities = item_similarities[rated_items]
    ratings = user_ratings[rated_items]
    
    # Select the N nearest neighbors
    top_neighbors = similarities.nlargest(N)
    top_ratings = ratings[top_neighbors.index]
    
    # Compute weighted average
    if top_neighbors.sum() > 0:
        prediction = np.dot(top_neighbors, top_ratings) / top_neighbors.sum()
    else:
        prediction = 0  # Default prediction if no neighbors exist

    return prediction


def predict_with_popularity(user_id, item_id, user_item_matrix, similarity_matrix, item_popularity, N=10, favor_popular=True):
    """
    Predict the rating with a weighting function that adjusts based on item popularity.
    `favor_popular=True` favors popular items; `favor_popular=False` penalizes them.
    """
    # Get the similarity scores for the target item
    item_similarities = similarity_matrix[item_id]
    
    # Get the user's ratings
    user_ratings = user_item_matrix.loc[user_id]
    
    # Combine similarities and ratings for non-zero ratings
    rated_items = user_ratings[user_ratings > 0].index
    similarities = item_similarities[rated_items]
    ratings = user_ratings[rated_items]
    
    # Adjust weights based on popularity
    popularity_weights = item_popularity[rated_items]
    if favor_popular:
        adjusted_weights = similarities * popularity_weights
    else:
        adjusted_weights = similarities / (popularity_weights + 1)  # Avoid division by zero
    
    # Select the N nearest neighbors
    top_neighbors = adjusted_weights.nlargest(N)
    top_ratings = ratings[top_neighbors.index]
    
    # Compute weighted average
    if top_neighbors.sum() > 0:
        prediction = np.dot(top_neighbors, top_ratings) / top_neighbors.sum()
    else:
        prediction = 0  # Default prediction if no neighbors exist

    return prediction



# Step 1: Load the dataset and prepare the User-Item matrix
# Make sure 'filtered_ratings.csv' is in the same directory
ratings = pd.read_csv("filtered_ratings.csv")

# Create the User-Item matrix
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

# Fill NaN values with 0 for similarity calculations
user_item_matrix_filled = user_item_matrix.fillna(0)

# Print the User-Item matrix
print("User-Item Matrix:")
print(user_item_matrix.head())

# Step 2a: Compute Cosine Similarity
cosine_sim = cosine_similarity(user_item_matrix_filled.T)
cosine_sim_df = pd.DataFrame(cosine_sim, index=user_item_matrix.columns, columns=user_item_matrix.columns)

print("\nCosine Similarity Matrix (sample):")
print(cosine_sim_df.iloc[:5, :5])  # Display a sample for readability

# Step 2b: Compute Pearson Correlation
# Pearson correlation works with NaN values, so we use the original matrix
pearson_sim = user_item_matrix.corr(method='pearson')

print("\nPearson Correlation Matrix (sample):")
print(pearson_sim.iloc[:5, :5])  # Display a sample for readability

# Check if the "csv" folder exists
folder_path = "csv"
if not os.path.exists(folder_path):
    print(f"Creating the folder: {folder_path}")
    os.makedirs(folder_path)

# Save the CSV files
print("Saving cosine similarity matrix...")
cosine_sim_path = os.path.join(folder_path, "cosine_similarity.csv")
cosine_sim_df.to_csv(cosine_sim_path, index=True)

print("Saving Pearson similarity matrix...")
pearson_sim_path = os.path.join(folder_path, "pearson_similarity.csv")
pearson_sim.to_csv(pearson_sim_path, index=True)

print(f"Cosine similarity matrix saved at: {cosine_sim_path}")
print(f"Pearson similarity matrix saved at: {pearson_sim_path}")
print("Process completed successfully!")

# Compute item popularity
item_popularity = ratings['movieId'].value_counts()

# Make predictions
pred_cosine = predict_weighted_average(user_id=1, item_id=2, user_item_matrix=user_item_matrix_filled, similarity_matrix=cosine_sim_df)
pred_popular = predict_with_popularity(user_id=1, item_id=2, user_item_matrix=user_item_matrix_filled, similarity_matrix=cosine_sim_df, item_popularity=item_popularity, favor_popular=True)
pred_unpopular = predict_with_popularity(user_id=1, item_id=2, user_item_matrix=user_item_matrix_filled, similarity_matrix=cosine_sim_df, item_popularity=item_popularity, favor_popular=False)

print(f"Predicted rating (cosine, N neighbors): {pred_cosine}")
print(f"Predicted rating (favor popular): {pred_popular}")
print(f"Predicted rating (favor unpopular): {pred_unpopular}")


