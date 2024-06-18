import pandas as pd

# Load the dataset
df = pd.read_csv('C:\\Users\\Ritesh\\Documents\\github\\mlp\\week1\\data\\V5.csv')

# Replace "?" with NaN
df.replace('?', pd.NA, inplace=True)

# Select odd indexed columns
odd_indexed_columns = df.iloc[:, ::2]

# Get the value in the 5th indexed column of the 100th indexed row
value = odd_indexed_columns.iloc[100, 4]

# Print the value
print("Value in the 5th indexed column of the 100th indexed row in the selected dataframe:", value)
