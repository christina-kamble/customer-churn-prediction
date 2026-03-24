#  Customer Churn Prediction - Telecom

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://streamlit.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red?logo=streamlit)](https://christina-churn-prediction.streamlit.app)

> **Business problem:** A telecom company loses ~26% of customers annually. Each churned customer costs £200–£400 to replace. Can we predict who will leave — before they do?

---

## 🎯 Project Summary

This project builds an end-to-end machine learning pipeline to predict customer churn using the IBM Telco Customer Churn dataset (7,043 customers, 21 features). It goes beyond model accuracy to deliver **actionable business insights** and **model explainability** using SHAP values.

**Best model: XGBoost — ROC-AUC ~0.82**

---

## 📁 Repository Structure

```
├── notebooks/
│   └── churn_analysis.ipynb      # Full EDA → preprocessing → modelling → insights
│
├── src/
│   ├── preprocess.py             # Reusable data cleaning & encoding pipeline
│   └── train.py                  # Model training, evaluation & persistence
│
├── dashboard/
│   └── app.py                    # Interactive Streamlit dashboard
│
├── outputs/                      # Saved plots and model artefacts
├── requirements.txt
└── README.md
```

---

## 🔍 Key Findings

| Finding | Churn Rate | Business Action |
|---|---|---|
| Month-to-month contracts | **42%** churn | Incentivise annual upgrades |
| Fibre optic internet | **41%** churn | Investigate service quality |
| No tech support | **41%** churn | Bundle support into plans |
| Tenure < 12 months | **~50%** churn | Launch onboarding loyalty programme |
| High monthly charges (>$65) | Higher churn | Introduce loyalty pricing tiers |

> Reducing churn by just 5% can increase profits by **25–95%** *(Harvard Business Review)*

---

## 📊 Model Comparison

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | ~0.84 | ~0.66 | ~0.55 | ~0.60 |
| Random Forest | ~0.82 | ~0.65 | ~0.47 | ~0.55 |
| **XGBoost** | **~0.82** | **~0.65** | **~0.52** | **~0.58** |

*XGBoost selected as best model for its balance of performance and explainability via SHAP.*

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/christina-kamble/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook notebooks/churn_analysis.ipynb

# 4. Launch the interactive dashboard
streamlit run dashboard/app.py
```

No data download needed — the dataset loads automatically from a public URL.

---

## 🧠 SHAP Explainability

This project uses **SHAP (SHapley Additive exPlanations)** to explain individual predictions — a critical skill for production ML and a key interview topic.

SHAP answers: *"Why did the model predict this customer will churn?"*

- Tenure: low tenure strongly increases churn probability
- Contract type: month-to-month contracts significantly increase risk
- Monthly charges: higher charges push predictions towards churn

---

## 🖥️ Interactive Dashboard

The Streamlit dashboard has three pages:

1. **Overview** — churn distribution, key drivers, business metrics
2. **Predict a Customer** — adjust sliders to simulate any customer profile and get an instant risk score with a recommendation
3. **Model Insights** — SHAP feature importance and beeswarm plots

---

## 📦 Dataset

**IBM Telco Customer Churn** — publicly available, widely used in industry interviews.

- 7,043 customers · 21 features · 26.5% churn rate
- Features include: contract type, tenure, monthly charges, internet service, payment method
- No download required — loaded directly via URL in all scripts

---

## 🔮 Future Improvements

- [ ] Hyperparameter tuning with Optuna
- [ ] Deploy dashboard to Streamlit Cloud (free hosting)
- [ ] Add SMOTE for class imbalance handling
- [ ] Survival analysis for time-to-churn modelling
- [ ] Add unit tests with pytest

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
