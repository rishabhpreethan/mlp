import pandas as pd

# Load the dataset
file_path = "mlp\\nppe1_preprocessing\\data\\NPPE1_Preprocessing1.csv"
dataset = pd.read_csv(file_path)

# Filter the dataset for houses near to exactly 6, 7, or 8 highways
houses_near_highways = dataset[dataset['HIGHWAYCOUNT'].isin([6, 7, 8])]

# Get the count of such houses
num_houses_near_highways = houses_near_highways.shape[0]
print(num_houses_near_highways)
