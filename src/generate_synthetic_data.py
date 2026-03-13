import numpy as np
import pandas as pd


def generate_synthetic_data(n_samples: int):
    np.random.seed(42)

    # 1. Generate synthetic pH values
    n_samples = 500
    pH = np.random.uniform(0, 14, n_samples)

    # 2. Define Universal Indicator Hue Anchors (0-360 degrees)
    # Mapping: 0(Red), 2(Orange), 4(Yellow), 7(Green), 10(Blue), 14(Purple)
    pH_anchors = [0, 2, 4, 7, 10, 14]
    hue_anchors = [0, 25, 45, 120, 210, 280]

    # Interpolate to get realistic hues
    hue = np.interp(pH, pH_anchors, hue_anchors)

    # Add natural variance (sensor noise/lighting shifts)
    hue += np.random.normal(0, 1.5, n_samples)
    hue = np.clip(hue, 0, 360) / 360.0  # Normalize for colorsys (0-1)

    # 3. Saturation and Value (High for vivid indicator colors)
    saturation = np.random.uniform(0.7, 0.95, n_samples)
    value = np.random.uniform(0.7, 0.95, n_samples)

    # Build dataframe
    df = pd.DataFrame({
        "Hue": hue*100,
        "Saturation": saturation*100,
        "Value": value*100,
        "pH": pH
    })

    df.to_csv("data/synthetic/ph_synthetic.csv", index=False)

    print("Synthetic dataset generated.")