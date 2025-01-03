import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

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
