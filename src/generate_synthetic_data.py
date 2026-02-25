import numpy as np
import pandas as pd

np.random.seed(42)

# Generate pH values
n_samples = 500
pH = np.random.uniform(1, 14, n_samples)

# Simulate nonlinear hue response
hue = 10 + 15 * pH - 0.5 * pH**2

# Add noise
hue += np.random.normal(0, 5, n_samples)

# Simulate saturation and value
saturation = np.random.uniform(60, 100, n_samples)
value = np.random.uniform(60, 100, n_samples)

# Build dataframe
df = pd.DataFrame({
    "Hue": hue,
    "Saturation": saturation,
    "Value": value,
    "pH": pH
})

df.to_csv("data/synthetic/ph_synthetic.csv", index=False)

print("Synthetic dataset generated.")