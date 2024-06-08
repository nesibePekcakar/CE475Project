import csv
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("project_data.csv")

# Extract features (x1, x2, x3, x4, x5, x6) and target variable (Y)
features = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
target = 'Y'

# Plot all data points in one graph
plt.figure(figsize=(10, 6))
for feature in features:
    plt.scatter(data[feature], data[target], label=feature)

plt.xlabel('Feature Values')
plt.ylabel('Y')
plt.title('Scatter Plot of Features against Y')
plt.legend()
plt.show()

for feature in features:
    plt.figure(figsize=(8, 6))
    plt.scatter(data[feature], data[target])
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.title(f'Scatter Plot of {feature} vs. {target}')
    plt.grid(True)
    plt.show()




