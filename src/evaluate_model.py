import matplotlib.pyplot as plt
from train_model import train
from sklearn.metrics import r2_score, mean_squared_error

y_test, predictions = train()

# Evaluate
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print(f"R² Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")

plt.scatter(y_test, predictions, alpha=0.5)
plt.xlabel("Actual pH")
plt.ylabel("Predicted pH")
plt.title("Actual vs Predicted pH")
plt.plot([1,14],[1,14])
plt.show()