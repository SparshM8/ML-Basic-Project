"""Convert UCI's IMDb labelled sentences into a clean CSV."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent
RAW_FILE = PROJECT_DIR / "raw" / "sentiment labelled sentences" / "imdb_labelled.txt"
OUTPUT_FILE = PROJECT_DIR / "data" / "imdb_reviews.csv"


def load_raw_reviews() -> pd.DataFrame:
    rows = []
    for line in RAW_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.rsplit("\t", 1)
        if len(parts) != 2:
            continue
        review, label = parts
        rows.append({"review": review, "sentiment": int(label)})
    return pd.DataFrame(rows)


def main() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw IMDb file not found at {RAW_FILE}")

    data = load_raw_reviews()
    train, test = train_test_split(
        data, test_size=0.20, random_state=42, stratify=data["sentiment"]
    )
    train = train.assign(split="train")
    test = test.assign(split="test")
    combined = pd.concat([train, test], ignore_index=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {len(combined)} movie-review sentences to {OUTPUT_FILE}")
    print(combined.groupby(["split", "sentiment"]).size().to_string())


if __name__ == "__main__":
    main()
