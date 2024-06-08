import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv("project_data.csv")

# Extract features (X) and target variable (y)
X = data[['x1', 'x3', 'x5']]
y = data['Y']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 150, 200, 250, 300, 350, 360, 370],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.25],
    'max_depth': [3, 4, 5]
}

# Initialize KFold
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize the Gradient Boosting Regressor
model = GradientBoostingRegressor()

# Initialize GridSearchCV with KFold
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error')

# Perform GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train a new Gradient Boosting model using the best hyperparameters
best_model = GradientBoostingRegressor(**best_params)
best_model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE with best hyperparameters: {mse}")

"""Best Hyperparameters: {'learning_rate': 0.23, 'max_depth': 4, 'n_estimators': 150}
Test MSE with best hyperparameters: 2129216.811221413"""

"""Best Hyperparameters: {'learning_rate': 0.25, 'max_depth': 4, 'n_estimators': 370}
Test MSE with best hyperparameters: 2114793.9487205464"""

"""Best Hyperparameters: {'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 350}
Test MSE with best hyperparameters: 2665738.3546443367"""
