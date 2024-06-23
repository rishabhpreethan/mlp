# What are the two most important features computed by RFE?
#     Preprocess the data using pipeline shown in the diagram. Use LogisticRegression (with default parameters) for the estimator. Encode target variable via ordinal encoding.


import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_selection import RFE, VarianceThreshold

# Load the dataset
file_path = 'mlp\\week3\\dataset\\DataPreprocessingGraded_dataset.csv'
df = pd.read_csv(file_path)

# Preprocess the data to handle missing values represented by np.nan
df.replace('?', np.nan, inplace=True)

# Separate features and target
X = df.drop(columns=['Target'])
y = df['Target']

# Encode target variable
target_encoder = OrdinalEncoder()
y_encoded = target_encoder.fit_transform(y.values.reshape(-1, 1)).ravel()

# Define the column transformer for numeric and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), [0, 1, 2, 3]),
        ('cat', Pipeline(steps=[
            ('encoder', OrdinalEncoder())
        ]), [4])
    ]
)

# Define the complete pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('variance_threshold', VarianceThreshold(threshold=0.1))
])

# Apply the pipeline
X_transformed = pipeline.fit_transform(X)

# Apply RFE with Logistic Regression to find the two most important features
model = LogisticRegression()
rfe = RFE(model, n_features_to_select=2)
fit = rfe.fit(X_transformed, y_encoded)

# Get the two most important features
important_features_indices = np.where(fit.support_)[0]
print(important_features_indices)