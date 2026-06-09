# bank-churn-prediction

---

##  Overview

This project builds a complete machine learning pipeline to predict whether a bank customer will churn (leave the bank). It covers everything from raw data cleaning to model evaluation and business interpretation — the goal being to identify high-risk customers before they leave.

**Churn rate in the dataset: ~20%** — making this a classic class imbalance problem.

---

##  Project Structure

```
bank-churn-prediction/
│
├── bank_churn_final.ipynb   # Full pipeline notebook
├── README.md
└── data/                    # Dataset (not included — see below)
```

---

##  Pipeline

```
Raw Data → Cleaning → Feature Engineering → EDA → Modeling → Evaluation → Insights
```

**1. Data Cleaning**
- Standardized column names and text values
- Removed duplicates and impossible values
- Fixed data types and encoded categorical features (One-Hot + Label Encoding)

**2. Feature Engineering**
- Engineered interaction features to improve signal
- Outlier detection and treatment

**3. Exploratory Data Analysis (EDA)**
- Target distribution analysis
- Hypothesis testing across key variables: age, balance, gender, geography, activity
- Correlation heatmap

**4. Modeling**
- Train/test split with stratification
- Class imbalance handled via **SMOTE** (Synthetic Minority Oversampling)
- StandardScaler applied before distance-based models

**5. Models Trained**
| Model | ROC-AUC | F1 (Churn class) |
|---|---|---|
| Dummy Baseline | 0.500 | 0.000 |
| K-Nearest Neighbors | ~0.76 | ~0.52 |
| Logistic Regression | ~0.77 | ~0.55 |
| **Random Forest** ✅ | **~0.86** | **~0.63** |

**6. Evaluation**
- Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
- ROC curve comparison across models
- Overfitting check (train vs test scores)
- 5-fold Cross-Validation
- Feature importance analysis

---

## 🔍 Key Findings

- **Age** is the strongest churn predictor — older customers churn significantly more
- **German customers** churn at nearly double the rate of French/Spanish customers
- Customers with **3–4 products** show extremely high churn rates
- **Inactive members** are far more likely to churn — engagement is a key retention lever
- Top features by importance: `age`, `numofproducts`, `estimatedsalary`, `creditscore`, `tenure`

---

##  Tech Stack

- **Language:** Python 3.12
- **Libraries:** pandas, numpy, scikit-learn, imbalanced-learn (SMOTE), matplotlib, seaborn

---

##  How to Run

1. Clone the repo
   ```bash
   git clone https://github.com/YOUR_USERNAME/bank-churn-prediction.git
   cd bank-churn-prediction
   ```

2. Install dependencies
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
   ```

3. Add the dataset to the `/data` folder (Kaggle: [Bank Customer Churn Dataset](https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction))

4. Open and run the notebook
   ```bash
   jupyter notebook bank_churn_final.ipynb
   ```

