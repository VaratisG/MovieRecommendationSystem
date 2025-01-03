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

# Step 3: Save the similarity matrices for later use
cosine_sim_df.to_csv("cosine_similarity.csv", index=True)
pearson_sim.to_csv("pearson_similarity.csv", index=True)

print("\nCosine and Pearson similarity matrices saved successfully.")
