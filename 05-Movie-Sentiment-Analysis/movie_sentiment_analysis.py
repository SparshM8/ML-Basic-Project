"""Movie Sentiment Analysis using TF-IDF and Logistic Regression.

The project uses Stanford's Large Movie Review Dataset (IMDb) and predicts
whether a review is positive or negative.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
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


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "imdb_reviews.csv"
PLOT_FILE = PROJECT_DIR / "confusion_matrix.png"


def load_data() -> pd.DataFrame:
    """Load the consolidated labeled IMDb review CSV."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_FILE}")

    data = pd.read_csv(DATA_FILE)
    required_columns = {"review", "sentiment", "split"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in dataset: {sorted(missing_columns)}")
    return data.dropna(subset=["review", "sentiment", "split"])


def save_confusion_matrix(y_test, predictions) -> None:
    """Save the sentiment confusion-matrix visualization."""
    matrix = confusion_matrix(y_test, predictions)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        xlabel="Predicted sentiment",
        ylabel="Actual sentiment",
        title="Confusion Matrix: IMDb Sentiment",
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
    train = data[data["split"] == "train"]
    test = data[data["split"] == "test"]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train["review"])
    X_test = vectorizer.transform(test["review"])
    y_train = train["sentiment"]
    y_test = test["sentiment"]

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    print("Movie Sentiment Analysis")
    print("=" * 25)
    print(f"Training reviews: {len(train)}")
    print(f"Testing reviews: {len(test)}")
    print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}\n")
    for name, value in metrics.items():
        print(f"{name.title():<10}: {value:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=["Negative", "Positive"]))

    print("Example prediction:")
    example = "The story was entertaining, beautifully acted, and surprisingly moving."
    example_prediction = model.predict(vectorizer.transform([example]))[0]
    print(f"  Review: {example}")
    print(f"  Sentiment: {'Positive' if example_prediction == 1 else 'Negative'}")

    save_confusion_matrix(y_test, predictions)


if __name__ == "__main__":
    main()

# Sources:
# https://ai.stanford.edu/~amaas/data/sentiment/
# https://huggingface.co/datasets/stanfordnlp/imdb
# https://aclanthology.org/P11-1015/
