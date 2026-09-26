# 📊 Telecom Customer Churn Prediction & CLV Analysis

An end-to-end machine learning and deep learning project that predicts telecom customer churn using an Artificial Neural Network (ANN), and combines churn probability with Customer Lifetime Value (CLV) to create a business-oriented customer retention prioritization system.

## 🎯 Project Overview

Telecom companies lose significant revenue when customers churn (cancel their service). This project builds a system to:
- Predict which customers are likely to churn, using an ANN supported by Logistic Regression and Random Forest baselines
- Estimate each customer's Customer Lifetime Value (CLV)
- Combine both into a Risk/Value segmentation framework to prioritize retention efforts
- Explain model predictions using SHAP
- Deliver predictions through an interactive Streamlit web app

## 📁 Dataset

[IBM Telco Customer Churn dataset](https://github.com/IBM/telco-customer-churn-on-icp4d) — 7,043 customers, 20 features (contract type, tenure, monthly charges, internet service, payment method, etc.) and a binary churn target.

## 🔧 Tech Stack

| Area | Technology |
|---|---|
| Data handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn (Logistic Regression, Random Forest) |
| Deep Learning | TensorFlow / Keras (ANN) |
| Explainability | SHAP |
| Deployment | Streamlit |

## 🧠 Model Performance

| Model | Accuracy | Recall (Churn) | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 80.6% | 56% | 0.842 |
| Random Forest | 78.9% | 50% | 0.828 |
| **ANN** | 78.0% | **61%** | 0.840 |

The ANN was selected as the primary model for deployment due to its higher recall on the churn class — prioritizing catching at-risk customers over raw accuracy, which better serves the retention use case.

## 🔑 Key Findings

- **Contract type, tenure, and monthly charges** are the strongest churn drivers
- Month-to-month customers churn at ~43%, vs. ~3% for two-year contracts
- Churn risk is highest in a customer's first few months and drops sharply with tenure
- Fiber optic customers, and customers without Online Security/Tech Support add-ons, churn more (identified via SHAP)

## 💰 CLV & Segmentation

CLV is estimated as `Average Monthly Charges × Expected Customer Lifetime (24 months)`. Customers are segmented into four groups based on churn risk and CLV, using median thresholds:

- **High Risk / High Value** — top retention priority
- **High Risk / Low Value** — lower priority
- **Low Risk / High Value** — nurture/retain
- **Low Risk / Low Value** — routine

## 🖥️ Streamlit App

An interactive app where a user enters a customer's attributes and receives:
- Predicted churn probability
- Estimated CLV
- Risk/Value segment classification

### Run locally

```bash
git clone https://github.com/Varaddahale/telecom-churn-clv-ann.git
cd telecom-churn-clv-ann
python -m venv venv
venv\Scripts\activate      # Windows
pip install streamlit tensorflow scikit-learn pandas numpy joblib
streamlit run app.py
```

## 📊 Project Workflow

Raw Data → Cleaning → EDA → Feature Engineering → Baseline Models → ANN → Evaluation → CLV Calculation → Segmentation → SHAP Explainability → Streamlit Deployment

## ⚠️ Limitations & Assumptions

- CLV formula uses a simplified 24-month expected lifetime assumption, not a validated industry standard
- SHAP findings represent model-learned associations, not proven causal relationships
- `tenure` and `TotalCharges` show high multicollinearity (0.83), which may affect Logistic Regression coefficient interpretation

## 👤 Author

Varad Dahale