from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Load the dataset
file_path = "mlp\\nppe1_preprocessing\\data\\NPPE1_Preprocessing1.csv"
dataset = pd.read_csv(file_path)

# Replace missing or unknown values with np.nan
# Assuming unknown values are non-numeric or specific placeholders
dataset['RM'] = pd.to_numeric(dataset['RM'], errors='coerce')
dataset['RIVERSIDE'] = pd.to_numeric(dataset['RIVERSIDE'], errors='coerce')
dataset['AGE'] = pd.to_numeric(dataset['AGE'], errors='coerce')

# Define features and target
features = dataset.drop(columns=['PRICE'])
target = dataset['PRICE']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=0)

# Get the number of samples in the training set
num_training_samples = X_train.shape[0]
print(num_training_samples)
