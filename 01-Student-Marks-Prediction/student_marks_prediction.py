"""Student Marks Prediction using Linear Regression.

This beginner project uses the UCI Student Performance dataset and predicts
G3, the final mathematics grade on a 0-20 scale.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "student-mat.csv"
PLOT_FILE = PROJECT_DIR / "actual_vs_predicted.png"

# G1 and G2 are the first- and second-period grades. They are intentionally
# included for this introductory example; the UCI documentation notes that
# they are strongly correlated with G3.
FEATURES = ["studytime", "failures", "absences", "G1", "G2"]
TARGET = "G3"


def load_data() -> pd.DataFrame:
    """Load the semicolon-separated UCI mathematics student dataset."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_FILE}. Download student-mat.csv first."
        )

    data = pd.read_csv(DATA_FILE, sep=";")
    required_columns = FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in data]
    if missing_columns:
        raise ValueError(f"Missing columns in dataset: {missing_columns}")
    return data[required_columns].dropna()


def train_and_evaluate(data: pd.DataFrame) -> None:
    """Train a linear regression model and report evaluation metrics."""
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("Student Marks Prediction")
    print("=" * 28)
    print(f"Rows used: {len(data)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} marks")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} marks")
    print(f"R-squared (R²): {r2:.2f}")
    print("\nModel coefficients:")
    for feature, coefficient in zip(FEATURES, model.coef_):
        print(f"  {feature}: {coefficient:.3f}")

    comparison = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
    print("\nSample predictions:")
    print(comparison.head(10).round(2).to_string(index=False))

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, predictions, alpha=0.75, color="#2563eb")
    minimum = min(y_test.min(), predictions.min())
    maximum = max(y_test.max(), predictions.max())
    plt.plot([minimum, maximum], [minimum, maximum], "--", color="#dc2626")
    plt.xlabel("Actual final grade (G3)")
    plt.ylabel("Predicted final grade")
    plt.title("Actual vs. Predicted Student Marks")
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    plt.close()
    print(f"\nChart saved to: {PLOT_FILE}")


if __name__ == "__main__":
    train_and_evaluate(load_data())

# Source: Cortez, P. (2008), UCI Student Performance dataset.
# https://doi.org/10.24432/C5TG7T
