"""House Price Prediction using the California Housing dataset.

The project compares Linear Regression with Random Forest Regression.
The target is measured in hundreds of thousands of US dollars.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DATA_FILE = DATA_DIR / "california_housing.csv"
PLOT_FILE = PROJECT_DIR / "actual_vs_predicted.png"

FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
TARGET = "MedHouseVal"


def load_or_download_data() -> pd.DataFrame:
    """Load the local CSV, or download the official dataset on first run."""
    if DATA_FILE.exists():
        return pd.read_csv(DATA_FILE)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = fetch_california_housing(as_frame=True)
    data = dataset.frame.copy()
    data.to_csv(DATA_FILE, index=False)
    print(f"Downloaded and saved dataset to: {DATA_FILE}")
    return data


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train one model, print metrics, and return its predictions."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)
    print(f"{name}")
    print(f"  MAE:  {mae:.3f} ($100,000 units; approximately ${mae * 100000:,.0f})")
    print(f"  RMSE: {rmse:.3f} ($100,000 units)")
    print(f"  R²:   {r2:.3f}")
    return predictions, r2


def main() -> None:
    data = load_or_download_data()
    missing_columns = [column for column in FEATURES + [TARGET] if column not in data]
    if missing_columns:
        raise ValueError(f"Missing columns in dataset: {missing_columns}")

    data = data[FEATURES + [TARGET]].dropna()
    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print("California House Price Prediction")
    print("=" * 36)
    print(f"Rows used: {len(data)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}\n")

    linear_predictions, linear_r2 = evaluate_model(
        "Linear Regression", LinearRegression(), X_train, X_test, y_train, y_test
    )
    forest_predictions, forest_r2 = evaluate_model(
        "Random Forest Regression",
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        X_train,
        X_test,
        y_train,
        y_test,
    )

    best_name = "Random Forest Regression" if forest_r2 >= linear_r2 else "Linear Regression"
    best_predictions = forest_predictions if forest_r2 >= linear_r2 else linear_predictions
    print(f"\nBetter R² on this split: {best_name}")

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, best_predictions, alpha=0.35, s=12, color="#2563eb")
    minimum = min(y_test.min(), best_predictions.min())
    maximum = max(y_test.max(), best_predictions.max())
    plt.plot([minimum, maximum], [minimum, maximum], "--", color="#dc2626")
    plt.xlabel("Actual median house value ($100,000 units)")
    plt.ylabel("Predicted median house value ($100,000 units)")
    plt.title(f"Actual vs. Predicted Values: {best_name}")
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    plt.close()
    print(f"Chart saved to: {PLOT_FILE}")


if __name__ == "__main__":
    main()

# Sources:
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
# https://scikit-learn.org/stable/datasets/real_world.html
