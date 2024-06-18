import pandas as pd

# Load the dataset
# Assuming the dataset is in a CSV file named 'data.csv'
df = pd.read_csv('C:\\Users\\Ritesh\\Documents\\github\\mlp\\week1\\data\\V5.csv')

# Access the value at the 692nd indexed row and 0th indexed column
value = df.iloc[692, 0]

print(f'The value at the 692nd indexed row and 0th indexed column is: {value}')
