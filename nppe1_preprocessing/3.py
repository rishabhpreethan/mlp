import pandas as pd

# Load the dataset to calculate the average house price
file_path = "mlp\\nppe1_preprocessing\\data\\NPPE1_Preprocessing1.csv"
dataset = pd.read_csv(file_path)

# Calculate the average house price
average_house_price = dataset['PRICE'].mean()
print(average_house_price)
