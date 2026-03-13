import matplotlib.pyplot as plt
import pandas as pd
from train_model import train
from sklearn.metrics import r2_score, mean_squared_error
from image_preprocessing import preprocess_image

y_test, predictions, model = train()

# Test sample images
hsv_from_image = preprocess_image("data/synthetic/sample_image.png")
print(f"Normalized HSV array from sample image: {hsv_from_image}")
input_df = pd.DataFrame(
    [hsv_from_image],
    columns=["Hue", "Saturation", "Value"]
)
prediction = model.predict(input_df)
print(f"pH Prediction from image: {prediction}")

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