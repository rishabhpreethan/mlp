import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = "mlp/nppe1_preprocessing/data/NPPE1_Preprocessing1.csv"
dataset = pd.read_csv(file_path)

# Drop CATEGORY column
if 'CATEGORY' in dataset.columns:
    dataset = dataset.drop(columns=['CATEGORY'])

# Define the columns to be scaled
min_max_columns = ['CRIM', 'ZN', 'POLINDEX', 'DIS', 'HIGHWAYCOUNT', 'TAX', 'PTRATIO', 'IMM', 'BPL']
standard_scale_columns = ['INDUS']

# Initialize the scalers and imputers
min_max_scaler = MinMaxScaler()
standard_scaler = StandardScaler()
median_imputer = SimpleImputer(strategy='median')
mean_imputer = SimpleImputer(strategy='mean')
most_frequent_imputer = SimpleImputer(strategy='most_frequent')

# Apply Min-Max scaling
dataset[min_max_columns] = min_max_scaler.fit_transform(dataset[min_max_columns])

# Apply Standard scaling
dataset[standard_scale_columns] = standard_scaler.fit_transform(dataset[standard_scale_columns])

# Impute RM with median and then apply Min-Max scaling
dataset['RM'] = median_imputer.fit_transform(dataset[['RM']])
dataset['RM'] = min_max_scaler.fit_transform(dataset[['RM']])

# Impute AGE with mean and then apply Min-Max scaling
dataset['AGE'] = mean_imputer.fit_transform(dataset[['AGE']])
dataset['AGE'] = min_max_scaler.fit_transform(dataset[['AGE']])

# Impute RIVERSIDE with the most frequent value and one-hot encode
dataset['RIVERSIDE'] = most_frequent_imputer.fit_transform(dataset[['RIVERSIDE']]).flatten()
dataset = pd.get_dummies(dataset, columns=['RIVERSIDE'], drop_first=True)

# Check the number of features after transformation
num_features_after_transformation = dataset.shape[1]
print(num_features_after_transformation)

# Define features and target
features = dataset.drop(columns=['PRICE'])
target = dataset['PRICE']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=0)

# Compute the mean of the transformed test feature matrix
mean_transformed_test_data = X_test.values.mean()
print(mean_transformed_test_data)
