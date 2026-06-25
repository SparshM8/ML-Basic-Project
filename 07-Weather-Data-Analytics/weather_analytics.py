"""Weather & Climate Time-Series Analytics.

This project analyzes 365 days of weather observation data using Pandas,
NumPy, and Matplotlib. It computes 7-day rolling statistics, monthly summary
aggregations, temperature trends, and precipitation dashboards.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "weather_data.csv"
PLOT_FILE = PROJECT_DIR / "weather_dashboard.png"


def load_and_preprocess() -> pd.DataFrame:
    """Load weather CSV and compute moving averages."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_FILE}")

    data = pd.read_csv(DATA_FILE)
    data["Date"] = pd.to_datetime(data["Date"])
    data["Month"] = data["Date"].dt.month_name()

    # Calculate 7-day rolling average for smooth temperature trends
    data["Temp_7D_Avg"] = data["Temperature_C"].rolling(window=7, min_periods=1).mean()
    return data


def generate_weather_dashboard(data: pd.DataFrame) -> None:
    """Compute meteorological summary statistics and render dashboard visualizer."""
    avg_temp = data["Temperature_C"].mean()
    max_temp = data["Temperature_C"].max()
    min_temp = data["Temperature_C"].min()
    total_rain = data["Rainfall_mm"].sum()
    rainy_days = (data["Rainfall_mm"] > 0).sum()

    print("Weather & Climate Analytics Report")
    print("=" * 35)
    print(f"Mean Annual Temp   : {avg_temp:.1f} °C")
    print(f"Maximum Temp Recorded: {max_temp:.1f} °C")
    print(f"Minimum Temp Recorded: {min_temp:.1f} °C")
    print(f"Total Rainfall      : {total_rain:.1f} mm")
    print(f"Total Rainy Days    : {rainy_days} days\n")

    # Monthly Summary Table
    monthly_agg = data.groupby("Month", sort=False).agg(
        Mean_Temp=("Temperature_C", "mean"),
        Total_Rain=("Rainfall_mm", "sum"),
        Avg_Humidity=("Humidity_Pct", "mean")
    ).round(1)

    print("Monthly Summary:")
    print(monthly_agg.to_string())

    # Multi-panel Dashboard Visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Annual Weather & Climate Analytics Dashboard", fontsize=16, fontweight="bold")

    # 1. Temperature & 7-day Moving Average
    axes[0].plot(data["Date"], data["Temperature_C"], color="#93c5fd", alpha=0.6, label="Daily Temp (°C)")
    axes[0].plot(data["Date"], data["Temp_7D_Avg"], color="#1d4ed8", linewidth=2, label="7-Day Moving Avg")
    axes[0].axhline(avg_temp, color="#dc2626", linestyle="--", label=f"Annual Mean ({avg_temp:.1f}°C)")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title("Daily Temperature & 7-Day Moving Average")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Daily Precipitation / Rainfall
    axes[1].bar(data["Date"], data["Rainfall_mm"], color="#0284c7", width=1.0)
    axes[1].set_ylabel("Rainfall (mm)")
    axes[1].set_title("Daily Rainfall Volume")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # 3. Humidity & Wind Speed Dual Trend
    axes[2].plot(data["Date"], data["Humidity_Pct"], color="#10b981", label="Humidity (%)", linewidth=1.5)
    axes[2].set_ylabel("Humidity (%)", color="#10b981")
    axes[2].tick_params(axis="y", labelcolor="#10b981")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    ax2 = axes[2].twinx()
    ax2.plot(data["Date"], data["WindSpeed_kmh"], color="#f59e0b", label="Wind Speed (km/h)", linewidth=1.2, linestyle=":")
    ax2.set_ylabel("Wind Speed (km/h)", color="#f59e0b")
    ax2.tick_params(axis="y", labelcolor="#f59e0b")
    axes[2].set_title("Humidity & Wind Speed Trends")

    plt.xlabel("Date")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(PLOT_FILE, dpi=150)
    plt.close()
    print(f"\nDashboard chart saved to: {PLOT_FILE}")


if __name__ == "__main__":
    generate_weather_dashboard(load_and_preprocess())
