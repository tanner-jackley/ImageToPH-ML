import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from train_model import train
from sklearn.metrics import r2_score, mean_squared_error

y_test, predictions, model = train()

"""
# Test sample images
hsv_from_image = preprocess_image("data/synthetic/sample_image.png")
input_df = pd.DataFrame(
    [hsv_from_image],
    columns=["Hue", "Saturation", "Value"]
)
prediction = model.predict(input_df)
print(f"pH Prediction from sample_image.png: {prediction}")

hsv_from_image = preprocess_image("data/synthetic/10.png")
input_df = pd.DataFrame(
    [hsv_from_image],
    columns=["Hue", "Saturation", "Value"]
)
prediction = model.predict(input_df)
print(f"pH Prediction from 10.png: {prediction}")

hsv_from_image = preprocess_image("data/synthetic/3.png")
input_df = pd.DataFrame(
    [hsv_from_image],
    columns=["Hue", "Saturation", "Value"]
)
prediction = model.predict(input_df)
print(f"pH Prediction from 3.png: {prediction}")
"""

# Evaluate
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print(f"R² Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")

y_test = np.array(y_test)
predictions = np.array(predictions).flatten()

zip(y_test, predictions)

plt.figure(figsize=(8,6))
plt.scatter(y_test, predictions, s=100, alpha=0.7)

# ideal line
plt.plot([0, 14], [0, 14], 'r--')

plt.xlabel("Actual pH")
plt.ylabel("Predicted pH")
plt.title("Actual vs Predicted pH")
plt.xlim(0, 14)
plt.ylim(0, 14)
plt.grid(True)
plt.show()