from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Load the dataset
housing = fetch_california_housing()

# Split the dataset into train and test sets with a 70:30 ratio
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.3, random_state=1)

# Scale the features
scaler = StandardScaler(with_mean=True, with_std=True)
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# Set up the parameter grid for GridSearchCV
param_grid = {
    'alpha': [0.5, 0.1, 0.05, 0.01, 0.005, 0.001],
    'fit_intercept': [True, False]
}

# Initialize the Ridge Regression model
ridge = Ridge()

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=4)

# Train the model
grid_search.fit(X_train_norm, y_train)

# Make predictions on the test set
y_pred = grid_search.predict(X_test_norm)

# Calculate the score (R^2 score)
score = grid_search.score(X_test_norm, y_test)

# Output the score
print(f"Score: {score:.4f}")

# Output the best parameters
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")
