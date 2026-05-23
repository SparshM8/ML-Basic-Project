"""Titanic Survival Prediction using binary classification.

This beginner project compares Logistic Regression and Decision Tree models
on a public Titanic passenger dataset.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
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
from sklearn.tree import DecisionTreeClassifier


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "titanic.csv"
PLOT_FILE = PROJECT_DIR / "confusion_matrix.png"
TARGET = "Survived"

FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "Siblings/Spouses Aboard",
    "Parents/Children Aboard",
    "Fare",
]


def load_data() -> pd.DataFrame:
    """Load the Titanic CSV and keep the beginner-friendly features."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_FILE}")

    data = pd.read_csv(DATA_FILE)
    required_columns = FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in data]
    if missing_columns:
        raise ValueError(f"Missing columns in dataset: {missing_columns}")
    return data[required_columns].copy()


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing for numeric and categorical passenger features."""
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
    """Train a classifier and print useful binary-classification metrics."""
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
    print(classification_report(y_test, predictions, target_names=["Did not survive", "Survived"], zero_division=0))
    return predictions, metrics


def save_confusion_matrix(y_test, predictions, model_name: str) -> None:
    """Save a readable confusion-matrix visualization."""
    matrix = confusion_matrix(y_test, predictions)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Did not survive", "Survived"],
        yticklabels=["Did not survive", "Survived"],
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
    X = data.drop(columns=[TARGET])
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("Titanic Survival Prediction")
    print("=" * 29)
    print(f"Rows used: {len(data)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Survival rate: {y.mean():.1%}\n")

    logistic_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    tree_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            ("model", DecisionTreeClassifier(max_depth=4, random_state=42)),
        ]
    )

    logistic_predictions, logistic_metrics = evaluate_model(
        "Logistic Regression", logistic_pipeline, X_train, X_test, y_train, y_test
    )
    tree_predictions, tree_metrics = evaluate_model(
        "Decision Tree", tree_pipeline, X_train, X_test, y_train, y_test
    )

    if tree_metrics["f1"] >= logistic_metrics["f1"]:
        best_name = "Decision Tree"
        best_predictions = tree_predictions
    else:
        best_name = "Logistic Regression"
        best_predictions = logistic_predictions
    print(f"Better F1 score on this split: {best_name}")
    save_confusion_matrix(y_test, best_predictions, best_name)


if __name__ == "__main__":
    main()

# Sources:
# https://www.openml.org/d/40945
# https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv
