"""Titanic Survival Prediction Baseline."""
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "titanic.csv"

def main():
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded Titanic dataset with {len(df)} passengers.")

if __name__ == "__main__":
    main()
