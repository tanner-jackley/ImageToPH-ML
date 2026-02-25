import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train():
    # Load data
    df = pd.read_csv("data/synthetic/ph_synthetic.csv")

    X = df[["Hue", "Saturation", "Value"]]
    y = df["pH"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    return y_test, predictions