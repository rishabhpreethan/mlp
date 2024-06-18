import pandas as pd

# Load the dataset
df = pd.read_csv('C:\\Users\\Ritesh\\Documents\\github\\mlp\\week1\\data\\V5.csv')

# Replace "?" with NaN
df.replace('?', pd.NA, inplace=True)

# Find features with missing values
features_with_missing_values = df.columns[df.isnull().any()].tolist()

# Print features with missing values
print("Features with missing values:")
print(features_with_missing_values)
