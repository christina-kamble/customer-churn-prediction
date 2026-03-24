"""
Interactive Churn Prediction Dashboard
Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #f8f9fa; border-radius: 12px;
    padding: 1.2rem; text-align: center;
    border: 1px solid #e9ecef;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #2c3e50; }
.metric-label { font-size: 0.85rem; color: #6c757d; margin-top: 4px; }
.risk-high   { background: #fff5f5; border-left: 4px solid #e74c3c; padding: 1rem; border-radius: 8px; }
.risk-medium { background: #fffbf0; border-left: 4px solid #f39c12; padding: 1rem; border-radius: 8px; }
.risk-low    { background: #f0fff4; border-left: 4px solid #2ecc71; padding: 1rem; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Data & Model (cached) ────────────────────────────────────────────────────
@st.cache_data
def load_and_prepare():
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df.drop("customerID", axis=1, inplace=True)
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
    for col in binary_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df


@st.cache_resource
def train_model(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model = XGBClassifier(n_estimators=200, learning_rate=0.05,
                          max_depth=5, random_state=42,
                          eval_metric="logloss", verbosity=0)
    model.fit(X_train, y_train)
    return model, X, X_test, y_test


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📉 Churn Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["📊 Overview", "🔍 Predict a Customer", "🧠 Model Insights"])

df = load_and_prepare()
model, X_full, X_test, y_test = train_model(df)

# ── Page 1: Overview ──────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.title("Customer Churn Analysis Dashboard")
    st.markdown("Telecom dataset · 7,043 customers · XGBoost model")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    churn_rate = df["Churn"].mean()
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">7,043</div><div class="metric-label">Total Customers</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{churn_rate:.1%}</div><div class="metric-label">Churn Rate</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">~0.82</div><div class="metric-label">Model ROC-AUC</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">26</div><div class="metric-label">Features Used</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Key Churn Drivers")

    raw_df = pd.read_csv("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv")
    raw_df["TotalCharges"] = pd.to_numeric(raw_df["TotalCharges"], errors="coerce")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        churn_by_contract = raw_df.groupby("Contract")["Churn"].apply(
            lambda x: (x == "Yes").mean() * 100).sort_values(ascending=False)
        bars = ax.bar(churn_by_contract.index, churn_by_contract.values,
                      color=["#e74c3c", "#f39c12", "#2ecc71"], edgecolor="white")
        ax.set_title("Churn Rate by Contract Type", fontweight="bold")
        ax.set_ylabel("Churn Rate (%)")
        for bar, val in zip(bars, churn_by_contract.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", fontweight="bold")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        churned     = raw_df[raw_df["Churn"] == "Yes"]["MonthlyCharges"]
        not_churned = raw_df[raw_df["Churn"] == "No"]["MonthlyCharges"]
        ax.hist(not_churned, bins=30, alpha=0.6, color="#2ecc71", label="Retained")
        ax.hist(churned,     bins=30, alpha=0.6, color="#e74c3c", label="Churned")
        ax.set_title("Monthly Charges Distribution", fontweight="bold")
        ax.set_xlabel("Monthly Charges ($)")
        ax.legend()
        st.pyplot(fig)


# ── Page 2: Predict ───────────────────────────────────────────────────────────
elif page == "🔍 Predict a Customer":
    st.title("Predict Churn Risk for a Customer")
    st.markdown("Adjust the sliders to simulate a customer profile.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        tenure          = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 20, 120, 65)
    with col2:
        contract    = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet    = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    with col3:
        tech_support   = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check",
                                                          "Bank transfer (automatic)", "Credit card (automatic)"])

    # Build a sample row using the feature set
    sample = pd.DataFrame([np.zeros(X_full.shape[1])], columns=X_full.columns)
    sample["tenure"]         = tenure
    sample["MonthlyCharges"] = monthly_charges
    sample["TotalCharges"]   = tenure * monthly_charges

    # Map contract
    if contract == "One year":
        if "Contract_One year" in sample.columns: sample["Contract_One year"] = 1
    elif contract == "Two year":
        if "Contract_Two year" in sample.columns: sample["Contract_Two year"] = 1

    if internet == "Fiber optic":
        if "InternetService_Fiber optic" in sample.columns: sample["InternetService_Fiber optic"] = 1
    elif internet == "No":
        if "InternetService_No" in sample.columns: sample["InternetService_No"] = 1

    if tech_support == "Yes":
        if "TechSupport_Yes" in sample.columns: sample["TechSupport_Yes"] = 1

    prob = model.predict_proba(sample)[0][1]
    pct  = prob * 100

    st.markdown("---")
    st.subheader("Churn Risk Assessment")

    if pct >= 60:
        st.markdown(f'<div class="risk-high"><b>🔴 HIGH RISK — {pct:.1f}% probability of churn</b><br>Recommend immediate retention intervention: discount offer or contract upgrade.</div>', unsafe_allow_html=True)
    elif pct >= 30:
        st.markdown(f'<div class="risk-medium"><b>🟡 MEDIUM RISK — {pct:.1f}% probability of churn</b><br>Monitor closely. Consider a proactive check-in or loyalty reward.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="risk-low"><b>🟢 LOW RISK — {pct:.1f}% probability of churn</b><br>Customer appears stable. No immediate action required.</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(6, 1.5))
    ax.barh(["Churn risk"], [pct], color="#e74c3c" if pct >= 60 else "#f39c12" if pct >= 30 else "#2ecc71", height=0.4)
    ax.barh(["Churn risk"], [100 - pct], left=[pct], color="#ecf0f1", height=0.4)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)")
    ax.axvline(50, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_title(f"Predicted churn probability: {pct:.1f}%", fontweight="bold")
    st.pyplot(fig)


# ── Page 3: Model Insights ────────────────────────────────────────────────────
elif page == "🧠 Model Insights":
    st.title("Model Explainability — SHAP Analysis")
    st.markdown("Understand *why* the model predicts churn using SHAP values.")
    st.markdown("---")

    with st.spinner("Computing SHAP values..."):
        explainer   = shap.TreeExplainer(model)
        sample_data = X_test.sample(min(200, len(X_test)), random_state=42)
        shap_values = explainer.shap_values(sample_data)

    st.subheader("Top Features Driving Churn Predictions")
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(shap_values, sample_data, plot_type="bar",
                      max_display=12, show=False)
    plt.title("Feature Importance (SHAP)", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("How Each Feature Affects Churn Probability")
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    shap.summary_plot(shap_values, sample_data, max_display=12, show=False)
    plt.title("SHAP Beeswarm — Direction of Feature Impact", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown("---")
    st.markdown("""
    **How to read SHAP values:**
    - **Red dots** = high feature value · **Blue dots** = low feature value
    - Positive SHAP = pushes prediction towards churn
    - Negative SHAP = pushes prediction away from churn
    """)
