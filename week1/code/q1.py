import pandas as pd
import numpy as np

# Load the dataset
# Assuming the dataset is in a CSV file named 'data.csv'
df = pd.read_csv('C:\\Users\\Ritesh\\Documents\\github\\mlp\\week1\\data\\V5.csv')

# Count the unknown values
unknown_count = (df == '?').sum().sum()
print(f'Total number of unknown ("?") values in the dataset: {unknown_count}')

# Replace unknown values with NaN
df.replace('?', np.nan, inplace=True)

# Save the cleaned dataset to a new CSV file if needed
# df.to_csv('cleaned_data.csv', index=False)

print(df)
