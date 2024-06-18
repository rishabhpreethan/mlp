from sklearn.datasets import load_breast_cancer

# Load the breast cancer dataset
data = load_breast_cancer()

# Count the number of benign (B) tumor cases
num_benign_cases = len(data.target[data.target == 1])

# Print the number of benign (B) tumor cases
print("Number of benign (B) tumor cases:", num_benign_cases)
