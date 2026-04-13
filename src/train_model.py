import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from image_preprocessing import preprocess_image
from pathlib import Path

def train():
    '''
    # Load data
    df = pd.read_csv("data/synthetic/ph_synthetic.csv")

    X = df[["Hue", "Saturation", "Value"]]
    y = df["pH"]
    '''

    folder_path = Path("data/meziani_images")
    df = []
    for image_path in folder_path.glob("*.png"):
        hsv_from_image = preprocess_image(image_path)
        row = pd.DataFrame(
            [[*hsv_from_image, image_path.stem]],
            columns=["Hue", "Saturation", "Value", "pH"]
        )
        df.append(row)

    df = pd.concat(df, ignore_index=True).sort_values(by="pH", key=lambda x: x.astype(float), ignore_index=True)

    X = df[["Hue", "Saturation", "Value"]]
    y = df["pH"].astype(float)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    #model = LinearRegression()
    #model = KNeighborsRegressor(n_neighbors=3)
    #model = DecisionTreeRegressor(random_state=0)
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    return y_test, predictions, model