import pandas as pd

# Load the dataset
df = pd.read_csv('C:\\Users\\Ritesh\\Documents\\github\\mlp\\week1\\data\\V5.csv')

# Replace "?" with NaN
df.replace('?', pd.NA, inplace=True)

# Select odd indexed rows
odd_indexed_rows = df.iloc[1::2]

# Get the value in the 3rd indexed column of the 332nd indexed row
value = odd_indexed_rows.iloc[331, 2]

# Print the value
print("Value in the 3rd indexed column of the 332nd indexed row in the selected dataframe:", value)
