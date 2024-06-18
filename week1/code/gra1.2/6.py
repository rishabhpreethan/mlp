import pandas as pd

# Load the dataset
df = pd.read_csv('C:\\Users\\Ritesh\\Documents\\github\\mlp\\week1\\data\\V5.csv')

# Replace "?" with NaN
df.replace('?', pd.NA, inplace=True)

# Convert 'Year' column to numeric
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Sort the dataset based on 'Year' column in descending order
df_sorted = df.sort_values('Year', ascending=False)

# Select the rows where 'Year' is in the six most recent years
recent_years_data = df_sorted[df_sorted['Year'].isin(df_sorted['Year'].unique()[:6])]

# Count the number of rows in the selected dataset
num_samples_recent_years = len(recent_years_data)

# Print the number of samples
print("Number of samples in the six most recent years:", num_samples_recent_years)
