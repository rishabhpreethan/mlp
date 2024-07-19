import pandas as pd

# Load the dataset
file_path = "mlp\\nppe1_preprocessing\\data\\NPPE1_Preprocessing1.csv"
dataset = pd.read_csv(file_path)

# Identify implausible values in the "RIVERSIDE" feature
implausible_riverside_values = dataset[(dataset['RIVERSIDE'] != 0) & (dataset['RIVERSIDE'] != 1)]
num_implausible_riverside_values = implausible_riverside_values.shape[0]

print(num_implausible_riverside_values)
