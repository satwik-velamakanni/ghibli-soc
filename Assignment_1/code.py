import numpy as np
import pandas as pd

# Load training and test datasets
train_data = pd.read_csv("train.csv")
test_data  = pd.read_csv("test.csv")

print(train_data.shape)
print(test_data.shape)

# Extract target column
target = train_data['SalePrice']

# Prepare training features (exclude categorical features and unwanted columns)
features_train = train_data.drop(columns=['Id', 'SalePrice']) \
                           .select_dtypes(exclude=['object'])

# Prepare test features (exclude categorical features and ID)
features_test = test_data.drop(columns=['Id']) \
                         .select_dtypes(exclude=['object'])

print(features_train.shape)
print(features_test.shape)
print(features_train)

# Count missing values in each numeric column
missing_info = pd.DataFrame({
    'column': features_train.columns,
    'missing_count': np.isnan(features_train.values).sum(axis=0)
})

print(missing_info[missing_info.missing_count > 0])

# Fill missing values using median imputation
from sklearn.impute import SimpleImputer

median_imputer = SimpleImputer(strategy='median')
train_filled = pd.DataFrame(
    median_imputer.fit_transform(features_train),
    columns=features_train.columns,
    index=features_train.index
)

# Standardize the features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_filled.values)

# Add bias term (intercept)
rows, cols = train_scaled.shape
X_final = np.c_[np.ones(rows), train_scaled]
y_final = target.values.reshape(-1, 1)

# Initialize weights
weights = np.zeros((cols + 1, 1))
learning_rate = 1e-3
iterations = 10000

# Gradient Descent optimization
for step in range(iterations):
    predictions = X_final.dot(weights)
    residuals = predictions - y_final
    gradients = (2 / rows) * X_final.T.dot(residuals)
    weights -= learning_rate * gradients

# Output coefficients
print("Intercept:", weights[0, 0])
print("First 5 coefs:", weights[1:6, 0])
