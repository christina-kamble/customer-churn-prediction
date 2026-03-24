"""
Model training, evaluation, and persistence for churn prediction.
Trains Logistic Regression, Random Forest, and XGBoost,
then saves the best model based on ROC-AUC.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from typing import Dict, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, classification_report,
                              roc_curve, ConfusionMatrixDisplay)
from xgboost import XGBClassifier


def build_models() -> Dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost":             XGBClassifier(n_estimators=200, learning_rate=0.05,
                                              max_depth=5, random_state=42,
                                              eval_metric="logloss", verbosity=0),
    }


def train_evaluate(models: Dict, X_train, X_test, X_train_sc, X_test_sc,
                   y_train, y_test) -> Dict:
    """Train all models and collect evaluation metrics."""
    results = {}
    for name, model in models.items():
        X_tr = X_train_sc if name == "Logistic Regression" else X_train
        X_te = X_test_sc  if name == "Logistic Regression" else X_test

        model.fit(X_tr, y_train)
        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        report = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            "model":   model,
            "y_pred":  y_pred,
            "y_proba": y_proba,
            "roc_auc": auc,
            "report":  report,
        }
        print(f"  {name:<25} ROC-AUC: {auc:.4f}  "
              f"F1(churn): {report['1']['f1-score']:.4f}")

    return results


def plot_roc_curves(results: Dict, y_test, save_path: str = "outputs/roc_curves.png"):
    """Plot and save ROC curves for all models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    colors = ["#3498db", "#2ecc71", "#e74c3c"]
    plt.figure(figsize=(8, 6))
    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})",
                 color=color, linewidth=2.5)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Model Comparison", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC curves saved to {save_path}")


def save_best_model(results: Dict, save_dir: str = "outputs"):
    """Save the best-performing model by ROC-AUC."""
    os.makedirs(save_dir, exist_ok=True)
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best = results[best_name]
    path = os.path.join(save_dir, "best_model.pkl")
    joblib.dump(best["model"], path)
    print(f"\nBest model: {best_name} (AUC={best['roc_auc']:.4f}) → saved to {path}")
    return best_name, best["model"]


def run_training():
    """Full training pipeline — loads preprocessed data, trains, evaluates, saves."""
    from src.preprocess import run_pipeline

    print("=== Running Preprocessing ===")
    X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, _ = run_pipeline()

    print("\n=== Training Models ===")
    models  = build_models()
    results = train_evaluate(models, X_train, X_test, X_train_sc, X_test_sc, y_train, y_test)

    plot_roc_curves(results, y_test)
    save_best_model(results)
    return results


if __name__ == "__main__":
    run_training()
