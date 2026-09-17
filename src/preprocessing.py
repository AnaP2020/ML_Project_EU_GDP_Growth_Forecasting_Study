"""
Preprocessing functions for the EU GDP growth forecasting project.
Each function does one thing and returns a DataFrame.
"""

import pandas as pd

from src import config


def load_clean_panel() -> pd.DataFrame:
    """Load the reshaped World Bank panel (837 rows, 27 countries, 1995-2025)."""
    df = pd.read_csv(config.CLEAN_DATA_PATH)
    df = df.sort_values([config.COUNTRY_CODE_COL, config.YEAR_COL]).reset_index(drop=True)
    return df


def add_gdp_growth_lags(df: pd.DataFrame) -> pd.DataFrame:
    """One-year and two-year lagged GDP growth, computed within each country."""
    df = df.copy()
    grouped = df.groupby(config.COUNTRY_CODE_COL)[config.TARGET_RAW_COL]
    df[config.LAG1_COL] = grouped.shift(1)
    df[config.LAG2_COL] = grouped.shift(2)
    return df


def add_next_year_target(df: pd.DataFrame) -> pd.DataFrame:
    """Next year's GDP growth, shifted within country."""
    df = df.copy()
    df[config.TARGET_COL] = (
        df.groupby(config.COUNTRY_CODE_COL)[config.TARGET_RAW_COL].shift(-1)
    )
    return df


def drop_unusable_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with no next-year target (2025 for each country)."""
    df = df.copy()
    before = len(df)
    df = df.dropna(subset=[config.TARGET_COL]).reset_index(drop=True)
    after = len(df)
    print(f"Dropped {before - after} rows with no target. Remaining: {after}")
    return df


def add_gdp_per_capita_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Assign each country to Lower / Middle / Higher GDP-per-capita tertile."""
    df = df.copy()
    country_mean = (
        df.groupby(config.COUNTRY_CODE_COL)["gdp_per_capita"]
        .mean()
        .reset_index()
        .rename(columns={"gdp_per_capita": "mean_gdp_per_capita"})
    )
    country_mean[config.GROUP_COL] = pd.qcut(
        country_mean["mean_gdp_per_capita"],
        q=3,
        labels=["Lower GDP per capita", "Middle GDP per capita", "Higher GDP per capita"],
    )
    df = df.merge(
        country_mean[[config.COUNTRY_CODE_COL, config.GROUP_COL]],
        on=config.COUNTRY_CODE_COL,
        how="left",
    )
    return df


def add_chronological_split(df: pd.DataFrame) -> pd.DataFrame:
    """Assign each row to 'train' (<=2019) or 'test' (>=2020)."""
    df = df.copy()
    df[config.SPLIT_COL] = "train"
    df.loc[df[config.YEAR_COL] >= config.TEST_START_YEAR, config.SPLIT_COL] = "test"
    return df


def impute_fdi_by_country_median(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing FDI with country-level median, fit on training rows only."""
    df = df.copy()
    train_medians = (
        df.loc[df[config.SPLIT_COL] == "train"]
        .groupby(config.COUNTRY_CODE_COL)["fdi_inflows_percent_gdp"]
        .median()
    )
    missing_mask = df["fdi_inflows_percent_gdp"].isna()
    df.loc[missing_mask, "fdi_inflows_percent_gdp"] = (
        df.loc[missing_mask, config.COUNTRY_CODE_COL].map(train_medians)
    )
    return df


def build_model_dataset() -> pd.DataFrame:
    """Full pipeline: load -> lags -> target -> drop -> groups -> split -> impute."""
    df = load_clean_panel()
    df = add_gdp_growth_lags(df)
    df = add_next_year_target(df)
    df = drop_unusable_rows(df)
    df = add_gdp_per_capita_groups(df)
    df = add_chronological_split(df)
    df = impute_fdi_by_country_median(df)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.MODEL_DATA_PATH, index=False)
    print(f"Saved {len(df)} rows to {config.MODEL_DATA_PATH}")
    return df