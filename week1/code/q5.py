import pandas as pd

# Load the dataset
df = pd.read_csv('C:\\Users\\Ritesh\\Documents\\github\\mlp\\week1\\data\\V5.csv')

# Replace "?" with NaN
df.replace('?', pd.NA, inplace=True)

# Find unique values in the Locality feature
unique_localities = df['Locality'].unique()

# Print the unique values
print("Unique values in the Locality feature:")
print(unique_localities)
