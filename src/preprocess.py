"""
Data preprocessing pipeline for telecom churn prediction.
Loads raw data, cleans it, encodes features, and saves train/test splits.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


def load_data(path: str = None) -> pd.DataFrame:
    """Load dataset from local path or remote URL."""
    source = path if path and os.path.exists(path) else DATA_URL
    df = pd.read_csv(source)
    print(f"Loaded {len(df):,} rows from {'local file' if path else 'remote URL'}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix data types, handle missing values, drop irrelevant columns."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df.drop("customerID", axis=1, inplace=True)
    print(f"After cleaning: {df.shape[1]} features, {df.isnull().sum().sum()} missing values")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode binary and categorical columns."""
    df = df.copy()
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService",
                   "PaperlessBilling", "Churn"]
    for col in binary_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    print(f"After encoding: {df.shape[1]} total columns")
    return df


def split_and_scale(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split into train/test and fit StandardScaler on training data."""
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test),      columns=X.columns)

    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    print(f"Churn rate — Train: {y_train.mean():.1%} | Test: {y_test.mean():.1%}")

    return X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, scaler


def run_pipeline(save_dir: str = "data/processed"):
    """Run full preprocessing pipeline and save outputs."""
    os.makedirs(save_dir, exist_ok=True)

    df = load_data()
    df = clean_data(df)
    df = encode_features(df)
    X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, scaler = split_and_scale(df)

    X_train.to_csv(f"{save_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{save_dir}/X_test.csv",  index=False)
    X_train_sc.to_csv(f"{save_dir}/X_train_scaled.csv", index=False)
    X_test_sc.to_csv(f"{save_dir}/X_test_scaled.csv",   index=False)
    y_train.to_csv(f"{save_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{save_dir}/y_test.csv",   index=False)
    joblib.dump(scaler, f"{save_dir}/scaler.pkl")

    print(f"\nAll files saved to {save_dir}/")
    return X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, scaler


if __name__ == "__main__":
    run_pipeline()
