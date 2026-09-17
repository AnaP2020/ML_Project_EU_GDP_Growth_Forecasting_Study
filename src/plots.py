"""Reusable plotting helpers."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid", context="talk")


def plot_feature_importance(importance_df, top_n=15, title="XGBoost feature importance"):
    """
    Horizontal bar chart of feature importances.
    importance_df: DataFrame with columns ['feature', 'importance']
    """
    df = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(df))))
    ax.barh(df["feature"], df["importance"], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax


def plot_country_mae(country_mae: pd.Series, title="Mean absolute error by country"):
    """
    Horizontal bar chart of MAE per country, sorted.
    country_mae: Series indexed by country code.
    """
    df = country_mae.sort_values()
    fig, ax = plt.subplots(figsize=(10, max(6, 0.3 * len(df))))
    ax.barh(df.index, df.values, color="indianred")
    ax.set_xlabel("MAE (percentage points)")
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax


def plot_group_mae(group_mae: pd.Series, title="MAE by GDP-per-capita group"):
    """Bar chart of MAE by GDP-per-capita group."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(group_mae.index, group_mae.values, color="seagreen")
    ax.set_ylabel("MAE (percentage points)")
    ax.set_title(title)
    plt.xticks(rotation=15)
    plt.tight_layout()
    return fig, ax


def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs predicted GDP growth"):
    """Scatter of actual vs predicted with a y=x reference line."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.5, color="steelblue", edgecolor="white")
    lims = [min(y_true.min(), y_pred.min()) - 1,
            max(y_true.max(), y_pred.max()) + 1]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual next-year GDP growth (%)")
    ax.set_ylabel("Predicted next-year GDP growth (%)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig, ax