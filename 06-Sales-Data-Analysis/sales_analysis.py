"""Retail Sales Analysis Baseline."""
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "sales_data.csv"

def main():
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} sales transactions.")

if __name__ == "__main__":
    main()
