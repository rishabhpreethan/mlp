import pandas as pd

# Load the dataset
file_path = "mlp\\nppe1_preprocessing\\data\\NPPE1_Preprocessing1.csv"
dataset = pd.read_csv(file_path)

# Identify implausible values in the "AGE" feature
implausible_age_values = dataset[(dataset['AGE'] < 0) | (dataset['AGE'] > 150)]
num_implausible_age_values = implausible_age_values.shape[0]

print(num_implausible_age_values)
