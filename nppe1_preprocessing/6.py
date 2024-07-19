import pandas as pd

# Load the dataset
file_path = "mlp\\nppe1_preprocessing\\data\\NPPE1_Preprocessing1.csv"
dataset = pd.read_csv(file_path)

# Identify implausible values for the number of rooms (RM feature)
# Let's assume implausible values are non-positive or unrealistically high values

implausible_values = dataset[(dataset['RM'] <= 0) | (dataset['RM'] > 20)]
num_implausible_values = implausible_values.shape[0]
print(num_implausible_values)
