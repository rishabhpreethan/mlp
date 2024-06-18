import pandas as pd

# Load the dataset
df = pd.read_csv('C:\\Users\\Ritesh\\Documents\\github\\mlp\\week1\\data\\V5.csv')

# Replace "?" with NaN
df.replace('?', pd.NA, inplace=True)

# Convert 'num_rooms' column to numeric
df['num_rooms'] = pd.to_numeric(df['num_rooms'], errors='coerce')

# Filter the dataset for the year 2022, Greenwich locality, num_rooms = 3, and facing North or East
filtered_data = df[(df['Year'] == 2022) & (df['Locality'] == 'Greenwich') & (df['num_rooms'] == 3) & ((df['direction_facing'] == 'North') | (df['direction_facing'] == 'East'))]

# Get the number of houses
num_houses = len(filtered_data)

# Print the number of houses
print("Number of houses in the year 2022, located in Greenwich, with exactly 3 rooms and facing North or East:", num_houses)
