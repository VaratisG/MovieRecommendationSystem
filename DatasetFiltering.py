import pandas as pd

# Load the ratings dataset
ratings = pd.read_csv("csv/ratings.csv")

# Preview the dataset
print("Dataset Preview:")
print(ratings.head())

# Filter movies with at least M ratings
M = 50
movie_counts = ratings['movieId'].value_counts()
filtered_movies = movie_counts[movie_counts >= M].index
ratings = ratings[ratings['movieId'].isin(filtered_movies)]

# Filter users with at least M' ratings
M_prime = 40  
user_counts = ratings['userId'].value_counts()
filtered_users = user_counts[user_counts >= M_prime].index
ratings = ratings[ratings['userId'].isin(filtered_users)]

# Count unique users and movies
num_users = ratings['userId'].nunique()
num_movies = ratings['movieId'].nunique()

# Print the counts
print(f"Number of unique users: {num_users}")
print(f"Number of unique movies: {num_movies}")

# Save the preprocessed data or preview it
print("Filtered Dataset Preview:")
print(ratings.head())

# Save the filtered dataset to a new CSV (optional)
ratings.to_csv("filtered_ratings.csv", index=False)