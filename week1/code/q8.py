import pandas as pd

# Load the dataset
df = pd.read_csv('C:\\Users\\Ritesh\\Documents\\github\\mlp\\week1\\data\\V5.csv')

# Replace "?" with NaN
df.replace('?', pd.NA, inplace=True)

# Drop samples with missing values strictly greater than 2
df_cleaned = df.dropna(thresh=df.shape[1]-2)

# Get the number of samples remaining
num_samples_remaining = len(df_cleaned)

# Print the number of samples remaining
print("Number of samples remaining after dropping samples with missing values greater than 2:", num_samples_remaining)
