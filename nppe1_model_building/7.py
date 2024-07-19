import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, SGDRegressor, Lasso
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
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
param_grid_sgd = {
    'penalty': ['l1', 'l2'],
    'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    'tol': [1e-4, 1e-3, 1e-2, 1e-1]
}

# Initialize the SGDRegressor
sgd_regressor = SGDRegressor(random_state=42)

# Initialize GridSearchCV
grid_search_sgd = GridSearchCV(estimator=sgd_regressor, param_grid=param_grid_sgd, cv=5, scoring='neg_mean_absolute_error')

# Fit GridSearchCV to the training data
grid_search_sgd.fit(X_train, y_train)

# Get the best parameters
best_params_sgd = grid_search_sgd.best_params_
print("Best parameters for SGDRegressor:", best_params_sgd)

# Use the best model from the GridSearchCV to predict on the test dataset
best_model_sgd = grid_search_sgd.best_estimator_
y_pred_test_sgd = best_model_sgd.predict(X_test)

# Calculate the mean absolute error on the test dataset
mae_test_sgd = mean_absolute_error(y_test, y_pred_test_sgd)
print("Mean Absolute Error on the test dataset using SGDRegressor:", mae_test_sgd)

# Define the pipeline
pipeline = Pipeline([
    ('pca', PCA()),
    ('lasso', Lasso(random_state=42))
])

# Define the parameter grid for GridSearchCV
param_grid_pipeline = {
    'pca__n_components': [0.9, 0.95],
    'lasso__alpha': [10, 1, 0.01, 0.001]
}

# Initialize GridSearchCV for the pipeline
grid_search_pipeline = GridSearchCV(estimator=pipeline, param_grid=param_grid_pipeline, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)

# Fit GridSearchCV to the training data
grid_search_pipeline.fit(X_train, y_train)

# Get the best parameters for the pipeline
best_params_pipeline = grid_search_pipeline.best_params_
print("Best parameters for pipeline:", best_params_pipeline)

# Use the best model from the GridSearchCV to predict on the test dataset
best_model_pipeline = grid_search_pipeline.best_estimator_
y_pred_test_pipeline = best_model_pipeline.predict(X_test)

# Calculate the R^2 score on the test dataset
r2_pipeline = r2_score(y_test, y_pred_test_pipeline)
print("R^2 score on the test dataset using the pipeline:", r2_pipeline)
