import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from sklearn.pipeline import Pipeline
import pickle
import warnings


warnings.filterwarnings('ignore')

# Define RMSE Scoring Function for Grid Search
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse, greater_is_better=False)

# Load dataset
Train_set = pd.read_csv("Train_set.csv")

# Target mean encoding
Train_set['manufactor_encoded'] = Train_set['manufactor'].map(Train_set.groupby('manufactor')['Price'].mean())
Train_set['car_model_encoded'] = Train_set['model'].map(Train_set.groupby('model')['Price'].mean())
Train_set['gear_encoded'] = Train_set['Gear'].map(Train_set.groupby('Gear')['Price'].mean())
Train_set['Engine_type_encoded'] = Train_set['Engine_type'].map(Train_set.groupby('Engine_type')['Price'].mean())
Train_set['prev_encoded'] = Train_set['Prev_ownership'].map(Train_set.groupby('Prev_ownership')['Price'].mean())
Train_set['curr_encoded'] = Train_set['Curr_ownership'].map(Train_set.groupby('Curr_ownership')['Price'].mean())

# Binning and encoding
bins = [0, 50000, 100000, 150000, 200000, float('inf')]
labels = ['0-50k', '50k-100k', '100k-150k', '150k-200k', '200k+']
Train_set['km_binned'] = pd.cut(Train_set['Km'], bins=bins, labels=labels)
Train_set['km_encoded'] = Train_set['km_binned'].map(Train_set.groupby('km_binned')['Price'].mean())

# Drop the original columns after encoding
Train_set = Train_set.drop(['manufactor', 'model', 'Prev_ownership', 'Curr_ownership', 'Gear', 'Engine_type', 'Km', 'km_binned'], axis=1)

X = Train_set.drop('Price', axis=1)  # Features
y = Train_set['Price']  # Target

# Split the data into training and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, random_state=42)

# Create a pipeline
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),  # Increased degree for better performance
    ('scaler', RobustScaler()),  # Robust to outliers
    ('elastic_net', ElasticNet(max_iter=20000, random_state=42))
])

# Parameter grid for Grid Search
param_grid = {
    'elastic_net__alpha': [0.01, 0.05, 0.1, 0.5, 1, 5, 10],  # Regularization strength
    'elastic_net__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],  # Balance between L1 and L2 regularization
}

# Setup GridSearchCV with 10-Fold Cross-Validation and RMSE Scoring
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1, scoring=rmse_scorer)

# Fit the model
grid_search.fit(X, y)

# Get the best model from GridSearchCV
best_model = grid_search.best_estimator_
print(f"Best Alpha: {grid_search.best_params_['elastic_net__alpha']}")
print(f"Best L1 Ratio: {grid_search.best_params_['elastic_net__l1_ratio']}")

# Predict on the test set
y_pred = best_model.predict(X)

# Set predictions to be non-negative
y_pred = np.maximum(y_pred, 0)

#rmse_value = np.sqrt(mean_squared_error(y, y_pred))
#print(f"RMSE for the best model: {rmse_value}")

# Save the model to a file
with open("trained_model.pkl", "wb") as f:
    pickle.dump(grid_search, f)