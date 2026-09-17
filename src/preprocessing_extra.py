"""
Loader and reshape for the additional World Bank indicators
(downloaded from WDI for potential inclusion in the model).

After data-availability analysis, only one indicator survives:
    BN.CAB.XOKA.GD.ZS  ->  current_account_percent_gdp

The other candidates were rejected due to insufficient coverage:
    FS.AST.PRVT.GD.ZS  (22.9% missing, structurally concentrated pre-2001)
    TT.PRI.MRCH.XD.WD  (35.5% missing, all pre-2005)
    FM.LBL.BMNY.GD.ZS  (74.8% missing)
"""

import pandas as pd

from src import config


EXTRA_DATA_PATH = config.RAW_DIR / "wdi_eu27_extra_indicators_1995_2025.csv"

# The single indicator we keep from the extra file
KEEP_CODE = "BN.CAB.XOKA.GD.ZS"
KEEP_COL  = "current_account_percent_gdp"

# Rename map for readability
CODE_TO_COL = {
    "BN.CAB.XOKA.GD.ZS": KEEP_COL,
}


def load_extra_panel() -> pd.DataFrame:
    """
    Load the raw extra-indicators CSV, drop World Bank metadata rows,
    reshape to long, and return a country-year DataFrame with only the
    kept indicators as columns.
    """
    raw = pd.read_csv(EXTRA_DATA_PATH)

    # Drop World Bank footer rows (NaN in Country Code)
    raw = raw.dropna(subset=["Country Code", "Series Code"]).copy()

    # Keep only the selected indicator
    raw = raw[raw["Series Code"].isin(CODE_TO_COL)].copy()

    # Identify year columns
    year_cols = [c for c in raw.columns if "[YR" in c]

    # Wide -> long
    long = raw.melt(
        id_vars=["Country Code", "Series Code"],
        value_vars=year_cols,
        var_name="Year",
        value_name="Value",
    )

    # Extract numeric year from "1995 [YR1995]" -> 1995
    long["Year"] = long["Year"].str.extract(r"(\d{4})").astype(int)

    # Coerce values to numeric (some are ".." or "" in World Bank exports)
    long["Value"] = pd.to_numeric(long["Value"], errors="coerce")

    # Rename Series Code -> friendly column name
    long["Column"] = long["Series Code"].map(CODE_TO_COL)

    # Pivot: one row per (Country Code, Year), one column per indicator
    wide = long.pivot_table(
        index=["Country Code", "Year"],
        columns="Column",
        values="Value",
        aggfunc="first",
    ).reset_index()

    wide.columns.name = None  # drop the pivot's column-axis name
    return wide

"""
Loader and merge for the additional World Bank indicators.

After data-availability analysis, only one indicator survives:
    BN.CAB.XOKA.GD.ZS  ->  current_account_percent_gdp

The other candidates were rejected due to insufficient coverage:
    FS.AST.PRVT.GD.ZS  (22.9% missing, structurally concentrated pre-2001)
    TT.PRI.MRCH.XD.WD  (35.5% missing, all pre-2005)
    FM.LBL.BMNY.GD.ZS  (74.8% missing)
"""

EXTRA_DATA_PATH = config.RAW_DIR / "wdi_eu27_extra_indicators_1995_2025.csv"

# The single indicator we keep from the extra file
CODE_TO_COL = {
    "BN.CAB.XOKA.GD.ZS": "current_account_percent_gdp",
}

# Lag to create for the current account
CURRENT_ACCOUNT_COL = "current_account_percent_gdp"
CURRENT_ACCOUNT_LAG_COL = "current_account_lag_1"


def load_extra_panel() -> pd.DataFrame:
    """
    Load the raw extra-indicators CSV, drop metadata rows, reshape to long,
    pivot to one row per (Country Code, Year), and return.
    """
    raw = pd.read_csv(EXTRA_DATA_PATH)

    # Drop World Bank footer rows
    raw = raw.dropna(subset=["Country Code", "Series Code"]).copy()
    raw = raw[raw["Series Code"].isin(CODE_TO_COL)].copy()

    year_cols = [c for c in raw.columns if "[YR" in c]

    long = raw.melt(
        id_vars=["Country Code", "Series Code"],
        value_vars=year_cols,
        var_name="Year",
        value_name="Value",
    )
    long["Year"] = long["Year"].str.extract(r"(\d{4})").astype(int)
    long["Value"] = pd.to_numeric(long["Value"], errors="coerce")
    long["Column"] = long["Series Code"].map(CODE_TO_COL)

    wide = long.pivot_table(
        index=["Country Code", "Year"],
        columns="Column",
        values="Value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def merge_extra_features(model_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join the extra indicators onto the model dataset, add a lagged
    current account, and impute any missing values with country-level
    medians computed from training rows only.
    """
    extra = load_extra_panel()

    df = model_df.merge(
        extra,
        on=["Country Code", "Year"],
        how="left",
    )

    # Lag the current account within country
    df = df.sort_values([config.COUNTRY_CODE_COL, config.YEAR_COL]).reset_index(drop=True)
    df[CURRENT_ACCOUNT_LAG_COL] = (
        df.groupby(config.COUNTRY_CODE_COL)[CURRENT_ACCOUNT_COL].shift(1)
    )

    # Impute current account missing values using country median (train only)
    train = df[config.SPLIT_COL] == "train"
    country_median = (
        df.loc[train]
        .groupby(config.COUNTRY_CODE_COL)[CURRENT_ACCOUNT_COL]
        .median()
    )
    mask = df[CURRENT_ACCOUNT_COL].isna()
    df.loc[mask, CURRENT_ACCOUNT_COL] = (
        df.loc[mask, config.COUNTRY_CODE_COL].map(country_median)
    )

    # Impute the lag column too, using country median of the lag
    country_median_lag = (
        df.loc[train]
        .groupby(config.COUNTRY_CODE_COL)[CURRENT_ACCOUNT_LAG_COL]
        .median()
    )
    mask_lag = df[CURRENT_ACCOUNT_LAG_COL].isna()
    df.loc[mask_lag, CURRENT_ACCOUNT_LAG_COL] = (
        df.loc[mask_lag, config.COUNTRY_CODE_COL].map(country_median_lag)
    )

    return df