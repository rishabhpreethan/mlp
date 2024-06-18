from sklearn.datasets import load_breast_cancer

# Load the breast cancer dataset
data = load_breast_cancer()

# Get the feature matrix shape
feature_matrix_shape = data.data.shape

# Print the feature matrix shape
print("Shape of the feature matrix:", feature_matrix_shape)
