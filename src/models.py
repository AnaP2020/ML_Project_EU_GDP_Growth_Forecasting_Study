"""
Model builders. Each builder returns an sklearn Pipeline with the
same interface, so we can swap models cleanly.

Pipeline structure:
    SimpleImputer  -> fills NaNs (lag_1/lag_2 have 27/54 NaNs)
    StandardScaler -> scales features (helps Linear Regression; harmless for trees)
    <Estimator>    -> Linear, Random Forest, or XGBoost

Why a Pipeline and not just a model?
  - Prevents data leakage: the imputer and scaler are fit *only* on training
    data during .fit(), then applied to test data during .predict().
  - Reproducible: the whole preprocessing + model is one object to save
    or re-train.
  - Cleaner notebooks: no juggling X_train_imputed_scaled everywhere.
"""

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from src import config


def build_linear_regression() -> Pipeline:
    """Baseline: impute -> scale -> OLS linear regression."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   LinearRegression()),
    ])


def build_random_forest() -> Pipeline:
    """Reference tree model: impute -> scale -> Random Forest."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        )),
    ])


def build_xgboost_baseline() -> Pipeline:
    """
    Flagship model — initial, untuned XGBoost.
    Reasonable defaults; tuning comes later in 05_hyperparameter_tuning.ipynb.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )),
    ])

class MeanBaseline(BaseEstimator, RegressorMixin):
    """
    Naive baseline: predicts the mean of the training target
    for every test row. This is the 'no information' model —
    any useful ML model must beat this.
    """
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class PersistenceBaseline(BaseEstimator, RegressorMixin):
    """
    Naive baseline: predicts next year's growth = this year's growth.
    Requires 'gdp_growth' to be the first feature, since that's the
    value we're persisting forward.
    """
    def fit(self, X, y):
        return self

    def predict(self, X):
        # X is a DataFrame; take the 'gdp_growth' column directly.
        return X["gdp_growth"].to_numpy()


def build_mean_baseline() -> MeanBaseline:
    return MeanBaseline()


def build_persistence_baseline() -> PersistenceBaseline:
    return PersistenceBaseline()