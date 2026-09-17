"""
Helpers for pulling the modeling data out of the rebuilt panel.

We keep this logic in one place so that every notebook and script
uses the *same* definition of features vs. target, and the same
train/test split.
"""

import pandas as pd

from src import config

from sklearn.model_selection import TimeSeriesSplit


# Columns that are NOT features for the model.
# They are either identifiers, bookkeeping, or (in the case of the
# GDP-per-capita group) reserved for post-model analysis.
NON_FEATURE_COLS = [
    config.COUNTRY_CODE_COL,
    config.COUNTRY_NAME_COL,
    config.YEAR_COL,
    config.TARGET_COL,       # this is what we predict
    config.GROUP_COL,        # for post-model segmentation only
    config.SPLIT_COL,        # train/test label
]

# The exact list of features the models will see.
FEATURE_COLS = (
    config.PREDICTOR_COLS
    + [config.LAG1_COL, config.LAG2_COL]
)


def get_train_test(df: pd.DataFrame):
    """
    Split the panel into train and test using the dataset_split column.

    Returns
    -------
    X_train, y_train, X_test, y_test : tuple of DataFrames / Series
        Features and targets for each split. Order matches the panel
        (sorted by country then year), which matters for TimeSeriesSplit.
    """
    train = df[df[config.SPLIT_COL] == "train"].copy()
    test  = df[df[config.SPLIT_COL] == "test"].copy()

    X_train = train[FEATURE_COLS]
    y_train = train[config.TARGET_COL]
    X_test  = test[FEATURE_COLS]
    y_test  = test[config.TARGET_COL]

    return X_train, y_train, X_test, y_test


def get_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the non-feature columns (country, year, group, split) so that
    after predictions we can join predictions back to their identifiers.

    This is how we'll compute per-group and per-country errors later.
    """
    return df[["Country Code", "Country Name", "Year",
               config.GROUP_COL, config.SPLIT_COL]].copy()



def get_time_series_cv(n_splits: int = 5) -> TimeSeriesSplit:
    """Chronological CV. Each fold's validation comes after its training."""
    return TimeSeriesSplit(n_splits=n_splits)