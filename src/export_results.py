import pandas as pd
import numpy as np
from pathlib import Path
from train_model import train

def export_results():
    # Run training
    y_test, predictions, model = train("rf")

    # Convert to numpy
    y_test = np.array(y_test)
    predictions = np.array(predictions).flatten()

    # Create DataFrame
    results_df = pd.DataFrame({
        "Actual_pH": y_test,
        "Predicted_pH": predictions
    })

    # OPTIONAL: sort for cleaner graphs
    results_df = results_df.sort_values(by="Actual_pH").reset_index(drop=True)

    # Create output directory
    output_dir = Path("../data/model_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save file
    file_path = output_dir / "random_forest_regressor.csv"
    results_df.to_csv(file_path, index=False)

    print(f"Results exported to: {file_path.resolve()}")

if __name__ == "__main__":
    export_results()