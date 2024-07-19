import pandas as pd

# Load the dataset
file_path = "mlp\\nppe1_preprocessing\\data\\NPPE1_Preprocessing1.csv"
dataset = pd.read_csv(file_path)

# Calculate the average price of the top 10 most expensive houses
top_10_expensive_houses = dataset.nlargest(10, 'PRICE')
average_price_top_10 = top_10_expensive_houses['PRICE'].mean()
print(average_price_top_10)
