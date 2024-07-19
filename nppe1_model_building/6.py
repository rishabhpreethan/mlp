import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

# Load the dataset
file_path = 'mlp\\nppe1_model_building\\data\\NPPE1_ModelBuilding3.csv'
data = pd.read_csv(file_path)

# Split the dataset into features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Ridge model
ridge_model = Ridge(alpha=10, solver='saga', tol=1e-4, random_state=42)
ridge_model.fit(X_train, y_train)

# Predict on the test dataset
y_pred = ridge_model.predict(X_test)

# Calculate the R^2 score
r2 = r2_score(y_test, y_pred)
print("R^2 score:", r2)

# Get the absolute values of the coefficients
coefficients = np.abs(ridge_model.coef_)

# Find the index of the most important feature
most_important_feature_index = np.argmax(coefficients)
print("Index of most important feature:", most_important_feature_index)

# Find the index of the least important feature
least_important_feature_index = np.argmin(coefficients)
print("Index of least important feature:", least_important_feature_index)

# Define the parameter grid for GridSearchCV
param_grid = {
    'penalty': ['l1', 'l2'],
    'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    'tol': [1e-4, 1e-3, 1e-2, 1e-1]
}

# Initialize the SGDRegressor
sgd_regressor = SGDRegressor(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=sgd_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error')

# Fit GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best parameters:", best_params)

# Use the best model from the GridSearchCV to predict on the test dataset
best_model = grid_search.best_estimator_
y_pred_test = best_model.predict(X_test)

# Calculate the mean absolute error on the test dataset
mae_test = mean_absolute_error(y_test, y_pred_test)
print("Mean Absolute Error on the test dataset:", mae_test)
