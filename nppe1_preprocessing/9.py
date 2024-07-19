import pandas as pd

# Load the dataset
file_path = "mlp\\nppe1_preprocessing\\data\\NPPE1_Preprocessing1.csv"
dataset = pd.read_csv(file_path)

# Remove rows with missing values in either 'RIVERSIDE' or 'AGE' columns
filtered_dataset = dataset.dropna(subset=['RIVERSIDE', 'AGE'])

# Filter the dataset for houses on riverside (RIVERSIDE = 1) and age 50 years or younger
riverside_and_young_houses = filtered_dataset[(filtered_dataset['RIVERSIDE'] == 1) & (filtered_dataset['AGE'] <= 50)]

# Get the count of such houses
num_riverside_and_young_houses = riverside_and_young_houses.shape[0]
print(num_riverside_and_young_houses)
