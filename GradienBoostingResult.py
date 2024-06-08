import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("project_data.csv")

# Extract features (x1, x3, x5) and target variable (Y)
X = data[['x1', 'x3', 'x5']]
y = data['Y']

# Define the Gradient Boosting regressor with best hyperparameters
gb = GradientBoostingRegressor(learning_rate=0.05, max_depth=5, n_estimators=350, random_state=42)

# Initialize KFold with 10 folds
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize lists to store evaluation metrics
mse_scores = []
r2_scores = []

# Perform k-fold cross-validation with all features
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the model
    gb.fit(X_train, y_train)

    # Make predictions
    y_pred = gb.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Append scores to lists
    mse_scores.append(mse)
    r2_scores.append(r2)

# Calculate mean MSE and R-squared
mean_mse = sum(mse_scores) / len(mse_scores)
mean_r2 = sum(r2_scores) / len(r2_scores)

print("Mean Squared Error (Cross-Validation) with all features: {:.2f}".format(mean_mse))
print("Mean R-squared (Cross-Validation) with all features: {:.2f}".format(mean_r2))

# Train the Gradient Boosting model using the entire training set with all features
gb.fit(X, y)

# If you have unknown data to predict, you can load it and make predictions
unknown_data = pd.read_csv("unknowns.csv")
unknown_data_selected = unknown_data[['x1', 'x3', 'x5']]

# Make predictions using the selected features
y_pred_unknown = gb.predict(unknown_data_selected)
print("Predictions on Unknown Data:")
print(y_pred_unknown)
"""Mean Squared Error (Cross-Validation) with all features: 1098960.08
Mean R-squared (Cross-Validation) with all features: 0.69
Predictions on Unknown Data:
[  915.44892757  2611.31201045   321.51714948 11894.82563022
   335.30298629   306.12483302    19.426137    6016.91661092
  4532.61468182  3585.7517102    -57.46581518  6473.00491753
   451.58835836  1264.70471031  2455.51769659  2796.69178816
  9503.18348686  3221.73590734  2721.48951241   567.56296405]"""



