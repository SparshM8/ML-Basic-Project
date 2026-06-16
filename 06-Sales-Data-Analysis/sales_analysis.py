"""Retail Sales Data Analysis & Visualization Dashboard.

This project performs exploratory data analysis (EDA) on retail transaction
data using Pandas, NumPy, and Matplotlib. It evaluates monthly revenue,
category performance, and payment preferences.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "sales_data.csv"
PLOT_FILE = PROJECT_DIR / "sales_dashboard.png"


def load_and_preprocess() -> pd.DataFrame:
    """Load sales CSV and parse datetime columns."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_FILE}")

    data = pd.read_csv(DATA_FILE)
    data["Date"] = pd.to_datetime(data["Date"])
    data["Month"] = data["Date"].dt.to_period("M")
    return data


def generate_sales_dashboard(data: pd.DataFrame) -> None:
    """Compute aggregate business metrics and render visualization dashboard."""
    # Monthly Total Revenue
    monthly_sales = data.groupby("Month")["TotalSales"].sum()
    monthly_sales.index = monthly_sales.index.astype(str)

    # Category Revenue Breakdown
    category_sales = data.groupby("Category")["TotalSales"].sum().sort_values(ascending=False)

    # Payment Method Share
    payment_counts = data["PaymentMethod"].value_counts()

    # Regional Revenue
    region_sales = data.groupby("Region")["TotalSales"].sum().sort_values(ascending=True)

    # Print Terminal Report
    total_revenue = data["TotalSales"].sum()
    total_orders = len(data)
    avg_order_value = data["TotalSales"].mean()

    print("Retail Sales Performance Report")
    print("=" * 35)
    print(f"Total Revenue     : ${total_revenue:,.2f}")
    print(f"Total Transactions: {total_orders}")
    print(f"Average Order Val : ${avg_order_value:.2f}\n")

    print("Top Revenue Categories:")
    for cat, val in category_sales.items():
        print(f"  {cat:<16}: ${val:,.2f}")

    # Plotting 2x2 Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Retail Sales Analytics Dashboard", fontsize=16, fontweight="bold")

    # 1. Monthly Revenue Trend Line
    axes[0, 0].plot(monthly_sales.index, monthly_sales.values, marker="o", color="#2563eb", linewidth=2)
    axes[0, 0].set_title("Monthly Revenue Trend")
    axes[0, 0].set_xlabel("Month")
    axes[0, 0].set_ylabel("Revenue ($)")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # 2. Revenue by Category Bar Chart
    colors = ["#1d4ed8", "#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe"]
    axes[0, 1].bar(category_sales.index, category_sales.values, color=colors)
    axes[0, 1].set_title("Total Revenue by Category")
    axes[0, 1].set_xlabel("Product Category")
    axes[0, 1].set_ylabel("Revenue ($)")
    axes[0, 1].tick_params(axis="x", rotation=25)

    # 3. Payment Method Distribution Pie Chart
    axes[1, 0].pie(payment_counts.values, labels=payment_counts.index, autopct="%1.1f%%", startangle=140, colors=["#10b981", "#3b82f6", "#f59e0b", "#ef4444"])
    axes[1, 0].set_title("Payment Method Share")

    # 4. Regional Revenue Horizontal Bar
    axes[1, 1].barh(region_sales.index, region_sales.values, color="#8b5cf6")
    axes[1, 1].set_title("Regional Sales Distribution")
    axes[1, 1].set_xlabel("Revenue ($)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(PLOT_FILE, dpi=150)
    plt.close()
    print(f"\nDashboard chart saved to: {PLOT_FILE}")


if __name__ == "__main__":
    generate_sales_dashboard(load_and_preprocess())
