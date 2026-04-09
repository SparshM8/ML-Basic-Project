"""Student Marks Prediction - Initial Data Loading Module."""
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "student-mat.csv"
FEATURES = ["studytime", "failures", "absences", "G1", "G2"]
TARGET = "G3"

def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_FILE, sep=";")
    return data[FEATURES + [TARGET]].dropna()

if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} rows.")
