"""Telco Customer Churn Prediction - Data Pipeline."""
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "Telco-Customer-Churn.csv"

def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_FILE)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    return data.dropna().copy()

if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} churn records.")
