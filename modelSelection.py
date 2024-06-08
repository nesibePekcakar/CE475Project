import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv("project_data.csv")

# Extract features (X) and target variable (y)
X = data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']]
y = data['Y']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Linear': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor()
}

# Dictionary to store mean MSE for each model
mean_mse = {}

# Evaluate models using cross-validation
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    mse_scores = -scores
    mean_mse[name] = mse_scores.mean()

# Plot MSE for each model
plt.figure(figsize=(8, 4))
plt.bar(mean_mse.keys(), mean_mse.values())
plt.xlabel('Model',fontsize=2)
plt.ylabel('Mean Squared Error (MSE)',fontsize=10)
plt.title('Mean Squared Error for Different Regression Models',fontsize=10)
plt.xticks(rotation=45, fontsize=7)
plt.yticks(fontsize=8)

plt.show()
