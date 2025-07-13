
---

# 📘 Credit Card Default Prediction — ML Project Documentation

## 📌 Project Overview

**Objective**:
Build an end-to-end ML pipeline to predict whether a credit card customer will default on their payment next month using demographic, financial, and historical billing/payment data.

**Dataset Source**:
Taiwan credit card client dataset (April 2005 to September 2005)
Contains: 30,000 records, 24 features + 1 target

---

## 📁 Project Structure

```
MLENDPROJECT/
│
├── data/
│   ├── rawdata.csv
│   ├── cleaned.csv
│   ├── x.csv
│   └── y.csv
│
├── models/
│   └── model.pkl                   # Best model (RandomForest)
│
├── notebooks/
│   ├── datacollection.ipynb       # Optional (if fetching or simulating data)
│   ├── datacleaning.ipynb         # Cleaning, outliers, missing values
│   ├── eda.ipynb                  # Exploratory Data Analysis
│   ├── modelbuilding.ipynb        # Model training & evaluation
│   └── preprocess.ipynb           # Final preprocessing before deployment
│
├── src/
│   ├── datacleaning.py
│   ├── datacollection.py
│   ├── modelbuilding.py
│   └── preprocess.py
│
├── app.py                         # Streamlit UI
├── requirements.txt
└── venv/                          # Virtual environment
```

---

## 🔍 Dataset Description

Each row represents a client. Key columns:

| Column                       | Description                                                                |
| ---------------------------- | -------------------------------------------------------------------------- |
| `LIMIT_BAL`                  | Credit limit (NT dollars)                                                  |
| `SEX`                        | Gender (1 = male, 2 = female)                                              |
| `EDUCATION`                  | (1 = grad school, 2 = university, 3 = high school, 4 = others)             |
| `MARRIAGE`                   | (1 = married, 2 = single, 3 = others)                                      |
| `AGE`                        | Age in years                                                               |
| `PAY_0`–`PAY_6`              | Repayment status in past 6 months (-1 = pay duly, 1 = 1 month delay, etc.) |
| `BILL_AMT1`–`BILL_AMT6`      | Monthly bill amount                                                        |
| `PAY_AMT1`–`PAY_AMT6`        | Monthly payment amount                                                     |
| `default.payment.next.month` | Target (1 = default, 0 = non-default)                                      |

---

## 🧼 Data Cleaning & Preprocessing

Performed in `datacleaning.ipynb`:

* Removed duplicates
* Converted datatypes
* Handled categorical encoding
* Normalized skewed features
* Addressed label imbalance (if any)

Preprocessing output saved to:

* `x.csv` — Cleaned features
* `y.csv` — Target values

---

## 📊 Exploratory Data Analysis (EDA)

Conducted in `eda.ipynb`:

* **Target Distribution**: \~22% defaulters, 78% non-defaulters
* **Correlations**: `LIMIT_BAL`, `PAY_0`, and `BILL_AMT1` were key predictors
* **Plots Used**:

  * Boxplots by `SEX`, `EDUCATION`
  * Histograms for `AGE`, `LIMIT_BAL`
  * Heatmaps for feature correlation

---

## 🤖 Model Building

Performed in `modelbuilding.ipynb`:

### Models Tried:

| Model               | Accuracy  |
| ------------------- | --------- |
| Logistic Regression | 80%       |
| Decision Tree       | 82%       |
| **Random Forest**   | **84%** ✅ |
| XGBoost             | 83%       |

**Best Model**: `RandomForestClassifier`
Saved to: `models/model.pkl`

### Final Metrics (Random Forest):

```
Accuracy  : 84%
Precision : 0.86
Recall    : 0.82
F1-Score  : 0.84
```

---

## 📱 Streamlit App (app.py)

### Features:

* User inputs via sliders and dropdowns
* Live prediction with `model.pkl`
* Displays success or risk warning

### How to Run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧪 Sample Input (from Streamlit)

```text
LIMIT_BAL: 20000
SEX: Female
EDUCATION: University
MARRIAGE: Single
AGE: 24
PAY_0 to PAY_6: [2, 2, -1, -1, -2, -2]
BILL_AMT1 to 6: [3913, 3102, 689, 0, 0, 0]
PAY_AMT1 to 6: [0, 689, 0, 0, 0, 0]
```

Prediction Output: 🔴 *High Risk of Default*

---

## ✅ Future Improvements

* Add SHAP/feature importance to explain predictions
* Batch prediction via CSV upload
* Deploy to cloud (Streamlit Share / Render / AWS)
* Address class imbalance with SMOTE or class weights
* Enable model retraining pipeline

---

## 📦 Dependencies (`requirements.txt`)

```txt
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
```

---

## 🔚 Summary

This project shows a complete ML pipeline from data loading and cleaning to model training, evaluation, and deployment using Streamlit. The Random Forest model performs well with an accuracy of **84%**, making it a reliable baseline for credit risk prediction tasks.

---


