import pandas as pd

# Load the dataset
df = pd.read_csv('C:\\Users\\Ritesh\\Documents\\github\\mlp\\week1\\data\\V5.csv')

# Replace "?" with NaN
df.replace('?', pd.NA, inplace=True)

# Select even indexed columns
even_indexed_columns = df.iloc[:, ::2]

# Select even indexed rows
even_indexed_rows = even_indexed_columns.iloc[::2]

# Get the value in the 3rd indexed column of the 255th indexed row
value = even_indexed_rows.iloc[254, 3]

# Print the value
print("Value in the 3rd indexed column of the 255th indexed row in the selected dataframe:", value)
