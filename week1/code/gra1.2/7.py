import pandas as pd

# Load the dataset
df = pd.read_csv('C:\\Users\\Ritesh\\Documents\\github\\mlp\\week1\\data\\V5.csv')

# Replace "?" with NaN
df.replace('?', pd.NA, inplace=True)

# Convert 'num_rooms' and 'num_bathrooms' columns to numeric
df['num_rooms'] = pd.to_numeric(df['num_rooms'], errors='coerce')
df['num_bathrooms'] = pd.to_numeric(df['num_bathrooms'], errors='coerce')

# Select rows where 'num_rooms' is 3 and 'num_bathrooms' is 3
filtered_data = df[(df['num_rooms'] == 3) & (df['num_bathrooms'] == 3)]

# Get the number of samples
num_samples = len(filtered_data)

# Print the number of samples
print("Number of samples where num_rooms = 3 and num_bathrooms = 3:", num_samples)
