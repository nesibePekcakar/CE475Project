import matplotlib.pyplot as plt

models = ['Random Forest', 'Gradient Boosting']
mse_values = [1354155.29, 1098960.08]
r2_values = [0.73, 0.69]

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Models')
ax1.set_ylabel('Mean Squared Error (MSE)', color=color)
ax1.bar(models, mse_values, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('R-squared (R²)', color=color)
ax2.plot(models, r2_values, color=color, marker='o')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Mean Squared Error (MSE) and R-squared (R²) Comparison')
plt.show()