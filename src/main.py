import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from image_preprocessing import preprocess_image
from train_model import train
from sklearn.metrics import r2_score, mean_squared_error
import argparse
import os
from pathlib import Path

y_test, predictions, model = train("rf")

# Set up argument parsing
parser = argparse.ArgumentParser(description="Process an image.")
parser.add_argument("--image", required=False, help="Path to the image file")
args = parser.parse_args()

# Check if file exists and print it
if args.image:
    if os.path.exists(args.image):
        print(f"Processing image: {args.image}")
        hsv_from_image = preprocess_image(args.image)
        input_df = pd.DataFrame(
            [hsv_from_image],
            columns=["Hue", "Saturation", "Value"]
        )
        prediction = model.predict(input_df)
        print(f"pH Prediction from {Path(args.image).stem}: {prediction}")
    else:
        print(f"Error: File {args.image} not found.")
else:
    print("No image provided. Skipping individual prediction and showing training graph...")

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