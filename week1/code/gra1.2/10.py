import pandas as pd

# Load the dataset
df = pd.read_csv('C:\\Users\\Ritesh\\Documents\\github\\mlp\\week1\\data\\V5.csv')

# Replace "?" with NaN
df.replace('?', pd.NA, inplace=True)

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Filter the dataset for the month of August
august_data = df[df['Date'].dt.month == 8]

# Get the number of samples
num_samples = len(august_data)

# Print the number of samples
print("Number of samples corresponding to the month of August across all years:", num_samples)
