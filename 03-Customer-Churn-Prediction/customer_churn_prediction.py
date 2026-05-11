"""Customer Churn Prediction using the IBM Telco Customer Churn dataset.

This beginner project compares Logistic Regression and Random Forest
classification after handling numeric and categorical features.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "Telco-Customer-Churn.csv"
PLOT_FILE = PROJECT_DIR / "confusion_matrix.png"
TARGET = "Churn"


def load_data() -> pd.DataFrame:
    """Load and lightly clean the IBM Telco Customer Churn CSV."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_FILE}")

    data = pd.read_csv(DATA_FILE)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data = data.dropna().copy()
    data[TARGET] = data[TARGET].map({"No": 0, "Yes": 1})
    return data


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing for numeric and categorical columns."""
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train a model, print classification metrics, and return predictions."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    print(name)
    print("-" * len(name))
    for metric_name, value in metrics.items():
        print(f"{metric_name.title():<10}: {value:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=["Stayed", "Churned"], zero_division=0))
    return model, predictions, metrics


def save_confusion_matrix(y_test, predictions, model_name: str) -> None:
    """Save a readable confusion-matrix visualization."""
    matrix = confusion_matrix(y_test, predictions)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Stayed", "Churned"],
        yticklabels=["Stayed", "Churned"],
        xlabel="Predicted label",
        ylabel="Actual label",
        title=f"Confusion Matrix: {model_name}",
    )
    threshold = matrix.max() / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                matrix[row, column],
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(PLOT_FILE, dpi=150)
    plt.close(figure)
    print(f"Confusion matrix saved to: {PLOT_FILE}")


def main() -> None:
    data = load_data()
    X = data.drop(columns=[TARGET, "customerID"])
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("Customer Churn Prediction")
    print("=" * 27)
    print(f"Rows used: {len(data)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Churn rate: {y.mean():.1%}\n")

    preprocessor = build_preprocessor(X)
    logistic_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    forest_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=150,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    _, logistic_predictions, logistic_metrics = evaluate_model(
        "Logistic Regression", logistic_pipeline, X_train, X_test, y_train, y_test
    )
    _, forest_predictions, forest_metrics = evaluate_model(
        "Random Forest", forest_pipeline, X_train, X_test, y_train, y_test
    )

    if forest_metrics["f1"] >= logistic_metrics["f1"]:
        best_name = "Random Forest"
        best_predictions = forest_predictions
    else:
        best_name = "Logistic Regression"
        best_predictions = logistic_predictions
    print(f"Better F1 score on this split: {best_name}")
    save_confusion_matrix(y_test, best_predictions, best_name)


if __name__ == "__main__":
    main()

# Sources:
# https://www.ibm.com/docs/en/cognos-analytics/12.1.x?topic=samples-telco-customer-churn
# https://github.com/IBM/telco-customer-churn-on-icp4d
