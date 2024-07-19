import pandas as pd

# Load the dataset
file_path = "mlp\\nppe1_preprocessing\\data\\NPPE1_Preprocessing1.csv"
dataset = pd.read_csv(file_path)

# Create a new column 'CATEGORY' based on the given conditions
dataset['CATEGORY'] = pd.cut(dataset['PRICE'], 
                             bins=[-float('inf'), 10, 20, 30, 40, float('inf')], 
                             labels=['1', '2', '3', '4', '5'])

# Find the category with the highest number of records
category_counts = dataset['CATEGORY'].value_counts()
highest_category = category_counts.idxmax()
print(highest_category, category_counts)
