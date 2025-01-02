import pandas as pd

# Load the dataset
df = pd.read_csv('ratings.csv')

# Display the first few rows
print(df.head())

# Check the column names and data types
print(df.info())