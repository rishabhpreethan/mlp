import pandas as pd

# Load the dataset
file_path = "mlp\\nppe1_preprocessing\\data\\NPPE1_Preprocessing1.csv"
dataset = pd.read_csv(file_path)

# Get the number of samples in the dataset
num_samples = dataset.shape[0]
print(num_samples)
