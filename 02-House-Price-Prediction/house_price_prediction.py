"""House Price Prediction - Baseline Linear Regression."""
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "california_housing.csv"

def main():
    data = pd.read_csv(DATA_FILE)
    print(f"Loaded California housing dataset with {len(data)} records.")

if __name__ == "__main__":
    main()
