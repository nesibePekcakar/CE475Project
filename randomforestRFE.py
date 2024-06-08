import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("project_data.csv")

# Extract features (x1, x2, x3, x4, x5, x6) and target variable (Y)
X = data[['x1', 'x3', 'x5',]]
y = data['Y']

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 5, 10, 15, 20, 25, 30],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8]
}

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize Random Forest regressor
rf = RandomForestRegressor(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error')

# Fit GridSearchCV
grid_search.fit(X, y)

# Get best parameters
best_params = grid_search.best_params_

# Initialize lists to store evaluation metrics
mse_scores = []
r2_scores = []

# Perform k-fold cross-validation with best parameters
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Create and train the Random Forest model with best parameters
    best_rf = RandomForestRegressor(**best_params, random_state=42)
    best_rf.fit(X_train, y_train)

    # Make predictions
    y_pred = best_rf.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Append scores to lists
    mse_scores.append(mse)
    r2_scores.append(r2)

# Calculate mean scores
mean_mse = sum(mse_scores) / len(mse_scores)
mean_r2 = sum(r2_scores) / len(r2_scores)

print("Best Parameters:", best_params)
print("Mean Squared Error (Cross-Validation): {:.2f}".format(mean_mse))
print("Mean R-squared (Cross-Validation): {:.2f}".format(mean_r2))

"""Best Parameters: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 150}
Mean Squared Error (Cross-Validation): 2302203.17  
Mean R-squared (Cross-Validation): 0.78

Process finished with exit code 0"""
