from sklearn.datasets import fetch_california_housing

# Load the California Housing dataset
housing_data = fetch_california_housing()

# Print the description of the dataset
print(housing_data.DESCR)

# Get the feature matrix shape
feature_matrix_shape = housing_data.data.shape

# Print the feature matrix shape
print("Shape of the feature matrix:", feature_matrix_shape)


# Get the labels of the first five attributes
first_five_labels = housing_data.feature_names[:5]

# Print the labels
print("Labels of the first five attributes:", first_five_labels)
