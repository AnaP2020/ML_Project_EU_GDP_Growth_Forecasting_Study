"""
Evaluation helpers. Computes metrics, builds comparison tables,
and produces per-group / per-country error analyses.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config


def compute_metrics(y_true, y_pred) -> dict:
    """Return MAE, RMSE, and R² in one dict."""
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2":   r2_score(y_true, y_pred),
    }


def evaluate_model(name: str, model, X_train, y_train, X_test, y_test) -> dict:
    """
    Fit a model and return its train and test metrics.
    Used to build the model comparison table.
    """
    model.fit(X_train, y_train)
    train_metrics = compute_metrics(y_train, model.predict(X_train))
    test_metrics  = compute_metrics(y_test,  model.predict(X_test))
    return {
        "model":      name,
        "train_MAE":  train_metrics["MAE"],
        "test_MAE":   test_metrics["MAE"],
        "train_RMSE": train_metrics["RMSE"],
        "test_RMSE":  test_metrics["RMSE"],
        "train_R2":   train_metrics["R2"],
        "test_R2":    test_metrics["R2"],
    }


def build_comparison_table(models_and_names, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """
    Given a list of (name, model) tuples, fit each and return a
    tidy DataFrame with one row per model.
    """
    rows = []
    for name, model in models_and_names:
        rows.append(evaluate_model(name, model, X_train, y_train, X_test, y_test))
    return pd.DataFrame(rows).set_index("model").round(3)