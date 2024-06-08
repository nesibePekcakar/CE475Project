import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("project_data.csv")

# Extract features (x1, x3, x5) and target variable (Y)
X = data[['x1', 'x3', 'x5']]
y = data['Y']

# Define the Random Forest regressor
rf = RandomForestRegressor(n_estimators=150, random_state=42)

# Initialize KFold with 10 folds
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize lists to store evaluation metrics
mse_scores = []
r2_scores = []

# Perform k-fold cross-validation with selected features
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the model
    rf.fit(X_train, y_train)

    # Make predictions
    y_pred = rf.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Append scores to lists
    mse_scores.append(mse)
    r2_scores.append(r2)

# Calculate mean scores
mean_mse = sum(mse_scores) / len(mse_scores)
mean_r2 = sum(r2_scores) / len(r2_scores)

print("Mean Squared Error (Cross-Validation) with selected features: {:.2f}".format(mean_mse))
print("Mean R-squared (Cross-Validation) with selected features: {:.2f}".format(mean_r2))

# Train the Random Forest model using the entire training set with selected features
rf.fit(X, y)

# If you have unknown data to predict, you can load it and make predictions
unknown_data = pd.read_csv("unknowns.csv")
y_pred_unknown = rf.predict(unknown_data[['x1', 'x3', 'x5']])
print("Predictions on Unknown Data:")
print(y_pred_unknown)
"""Mean Squared Error (Cross-Validation) with selected features: 1354155.29
Mean R-squared (Cross-Validation) with selected features: 0.73
Predictions on Unknown Data:
[  992.65333333  2486.22666667   446.56       11442.67333333
  1012.96         506.91333333    44.76666667  5420.24
  4184.79333333  3637.18666667   -22.92666667  6721.91333333
   119.25333333  1580.02        2204.17333333  3105.42666667
  9367.41333333  3743.84         973.16        1194.41333333]"""
