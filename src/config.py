"""
Central configuration for the EU GDP growth forecasting project.
Holds paths, column names, split boundaries, and constants.
"""

from pathlib import Path

SEED=8

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
# config.py lives in src/, so project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Input file: the reshaped panel we verified earlier
CLEAN_DATA_PATH = PROCESSED_DIR / "clean_wdi_eu27_1995_2025.csv"

# Output file: the rebuilt, model-ready dataset
MODEL_DATA_PATH = PROCESSED_DIR / "model_data_v2.csv"

# ---------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------
COUNTRY_CODE_COL = "Country Code"
COUNTRY_NAME_COL = "Country Name"
YEAR_COL = "Year"
TARGET_RAW_COL = "gdp_growth"
TARGET_COL = "target_gdp_growth_next_year"

# Continuous predictors (used as-is, year t values)
PREDICTOR_COLS = [
    "gdp_growth",              # current year's growth (most recent info)
    "gdp_per_capita",
    "inflation",
    "gross_fixed_capital_formation",
    "unemployment",
    "trade_openness",
    "fdi_inflows_percent_gdp",
    "government_consumption_percent_gdp",
    "population_growth",
]

# GDP growth lags (created by us)
LAG1_COL = "gdp_growth_lag_1"
LAG2_COL = "gdp_growth_lag_2"

# Auxiliary columns
GROUP_COL = "gdp_per_capita_group"
SPLIT_COL = "dataset_split"

# ---------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------
# Train: 1995-2019. Test: 2020-2024 (held out until final evaluation).
# Inside train, we'll use TimeSeriesSplit for cross-validation during tuning.
TRAIN_END_YEAR = 2019
TEST_START_YEAR = 2020

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
RANDOM_STATE = SEED