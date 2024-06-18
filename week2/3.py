from sklearn.datasets import load_breast_cancer

# Load the breast cancer dataset
data = load_breast_cancer()

# Count the number of malignant (M) tumor cases
num_malignant_cases = len(data.target[data.target == 0])

# Print the number of malignant (M) tumor cases
print("Number of malignant (M) tumor cases:", num_malignant_cases)
