"""Weather Analytics Time-Series Baseline."""
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "weather_data.csv"

def main():
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} daily weather records.")

if __name__ == "__main__":
    main()
