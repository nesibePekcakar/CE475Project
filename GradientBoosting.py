import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE

# Load the dataset
data = pd.read_csv("project_data.csv")

# Extract features (x1, x2, x3, x4, x5, x6) and target variable (Y)
X = data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']]
y = data['Y']

# Define the Gradient Boosting regressor with best hyperparameters
gb = GradientBoostingRegressor(learning_rate=0.2, max_depth=4, n_estimators=350, random_state=42)

# Initialize RFE with the Gradient Boosting regressor
rfe = RFE(estimator=gb, n_features_to_select=3)  # Select top 3 features

# Fit RFE
rfe.fit(X, y)

# Get the selected features
selected_features = X.columns[rfe.support_]

print("Selected features using RFE:")
print(selected_features)

# Use the selected features for further model training and evaluation
X_selected = X[selected_features]

# Initialize KFold with 10 folds
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize lists to store evaluation metrics
mse_scores = []
r2_scores = []

# Perform k-fold cross-validation with selected features
for train_index, test_index in kf.split(X_selected, y):
    X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
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

# Calculate mean scores
mean_mse = sum(mse_scores) / len(mse_scores)
mean_r2 = sum(r2_scores) / len(r2_scores)

print("Mean Squared Error (Cross-Validation) with selected features: {:.2f}".format(mean_mse))
print("Mean R-squared (Cross-Validation) with selected features: {:.2f}".format(mean_r2))

# Train the Gradient Boosting model using the entire training set with selected features
gb.fit(X_selected, y)

# If you have unknown data to predict, you can load it and make predictions
unknown_data = pd.read_csv("unknowns.csv")
y_pred_unknown = gb.predict(unknown_data[selected_features])
print("Predictions on Unknown Data:")
print(y_pred_unknown)
