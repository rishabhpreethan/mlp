import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_selection import VarianceThreshold

# Load the dataset
file_path = 'mlp\\week3\\dataset\\DataPreprocessingGraded_dataset.csv'
df = pd.read_csv(file_path)

# Preprocess the data to handle missing values represented by np.nan
df.replace('?', np.nan, inplace=True)

# Separate features and target
X = df.drop(columns=['Target'])
y = df['Target']

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

# Get the number of features after transformation
num_features = X_transformed.shape[1]
print(num_features)
