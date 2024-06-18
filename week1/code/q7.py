import pandas as pd

# Load the dataset
df = pd.read_csv('C:\\Users\\Ritesh\\Documents\\github\\mlp\\week1\\data\\V5.csv')

# Replace "?" with NaN
df.replace('?', pd.NA, inplace=True)

# Count missing values in each feature
missing_values_count = df.isnull().sum()

# Find the feature with the most missing values
feature_with_most_missing_values = missing_values_count.idxmax()

# Print the feature with the most missing values
print("Feature with the most missing values:", feature_with_most_missing_values)
