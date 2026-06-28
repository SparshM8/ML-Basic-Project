# 🤖 Machine Learning & Data Analytics Basic Portfolio Projects

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg)]()

Welcome to the **Machine Learning & Data Analytics Projects** repository! This collection showcases 7 end-to-end projects spanning regression analysis, binary classification, tabular feature engineering, decision trees, exploratory data analysis (EDA), time-series analytics, and natural language processing (NLP).

---

## 🚀 Projects Overview

| # | Project Name | Task / Domain | Primary Dataset | Tech & Algorithms | Key Metric |
|---|---|---|---|---|---|
| **01** | [Student Marks Prediction](./01-Student-Marks-Prediction) | Regression / Education | UCI Student Performance | Linear Regression | **$R^2$ = 0.78** (MAE: 1.34) |
| **02** | [California House Price Prediction](./02-House-Price-Prediction) | Regression / Real Estate | California Housing | Linear Reg vs. Random Forest | **$R^2$ = 0.805** (MAE: $32.7k) |
| **03** | [Telco Customer Churn Prediction](./03-Customer-Churn-Prediction) | Imbalanced Classification | IBM Telco Churn | Logistic Reg (Balanced) vs. Random Forest | **Recall = 79.7%** (ROC-AUC: 0.835) |
| **04** | [Titanic Survival Prediction](./04-Titanic-Survival-Prediction) | Classification / Historical | Kaggle / OpenML Titanic | Logistic Reg vs. Decision Tree | **Accuracy = 80.3%** (F1: 0.720) |
| **05** | [Movie Review Sentiment Analysis](./05-Movie-Sentiment-Analysis) | NLP Sentiment Analysis | Stanford IMDb Reviews | TF-IDF + Logistic Regression | **Accuracy = 76.0%** (ROC-AUC: 0.845) |
| **06** | [Retail Sales Data Analysis](./06-Sales-Data-Analysis) | EDA & Business Dashboards | Retail Sales Transactions | Pandas, NumPy, Matplotlib | **$749.1k Total Revenue** |
| **07** | [Weather Time-Series Analytics](./07-Weather-Data-Analytics) | Time-Series & Climate EDA | Daily Weather Observations | Pandas, NumPy, Matplotlib | **365-Day Rolling Temp & Rain** |

---

## 📁 Repository Directory Structure

```text
ML-Basic-Project/
│
├── 01-Student-Marks-Prediction/
│   ├── data/student-mat.csv
│   ├── student_marks_prediction.py
│   ├── actual_vs_predicted.png
│   ├── requirements.txt
│   └── README.md
│
├── 02-House-Price-Prediction/
│   ├── data/california_housing.csv
│   ├── house_price_prediction.py
│   ├── actual_vs_predicted.png
│   ├── requirements.txt
│   └── README.md
│
├── 03-Customer-Churn-Prediction/
│   ├── data/Telco-Customer-Churn.csv
│   ├── customer_churn_prediction.py
│   ├── confusion_matrix.png
│   ├── requirements.txt
│   └── README.md
│
├── 04-Titanic-Survival-Prediction/
│   ├── data/titanic.csv
│   ├── titanic_survival.py
│   ├── confusion_matrix.png
│   ├── requirements.txt
│   └── README.md
│
├── 05-Movie-Sentiment-Analysis/
│   ├── data/imdb_reviews.csv
│   ├── movie_sentiment_analysis.py
│   ├── prepare_dataset.py
│   ├── confusion_matrix.png
│   ├── requirements.txt
│   └── README.md
│
├── 06-Sales-Data-Analysis/
│   ├── data/sales_data.csv
│   ├── sales_analysis.py
│   ├── sales_dashboard.png
│   ├── requirements.txt
│   └── README.md
│
├── 07-Weather-Data-Analytics/
│   ├── data/weather_data.csv
│   ├── weather_analytics.py
│   ├── weather_dashboard.png
│   ├── requirements.txt
│   └── README.md
│
├── .gitignore
└── README.md
```

---

## 🛠️ Environment Setup & Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/SparshM8/ML-Basic-Project.git
cd ML-Basic-Project
```

### 2. Install Dependencies
Install core dependencies for all 7 projects:
```bash
pip install pandas scikit-learn matplotlib numpy
```

---

## 📊 Summary of Project Learnings

1. **Exploratory Data Analysis & Cleaning**: Handled missing values, encoded categorical variables, transformed raw text into TF-IDF numerical vectors.
2. **Time-Series & Business Analytics**: Applied rolling averages, grouped time aggregations, and rendered multi-panel Matplotlib dashboards.
3. **Model Selection & Benchmarking**: Evaluated parametric baseline models (Linear/Logistic Regression) against non-parametric tree & ensemble models (Decision Trees, Random Forests).
4. **Evaluation Metrics**: Selected domain-tailored metrics such as MAE/RMSE for house price regression, $R^2$ for marks prediction, Recall for churn targeting, and F1-score/ROC-AUC for class-imbalanced datasets.

---

## 📝 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
