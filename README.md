# Forecasting Economic Growth in the European Union

**A Comparative Machine Learning Study Using World Bank Indicators**

Predicting next-year GDP growth across the 27 EU member states using historical World Bank development indicators, comparing Linear Regression, Random Forest, and XGBoost against naive baselines.

---

## Project Overview

Economic growth is a key signal for governments, investors, and international organisations. Yet next-year GDP growth is notoriously difficult to forecast: it is driven by structural factors, cyclical dynamics, and unpredictable shocks (financial crises, pandemics, geopolitical events).

This project investigates whether machine learning can predict next-year GDP growth across the EU-27 using World Bank indicators from 1995 to 2024. The analysis compares several regression models, applies hyperparameter tuning, investigates which features matter most, and evaluates whether predictive accuracy varies across economies at different levels of development.

The **primary finding** is that no ML model reliably outperforms a naive mean baseline on held-out data. This is consistent with decades of empirical forecasting literature and represents the project's main contribution: a rigorous, honest investigation of the limits of ML for short-horizon macroeconomic prediction.

---

## Research Questions

1. How accurately can next-year GDP growth across EU countries be predicted using historical World Bank indicators?
2. Does XGBoost outperform a simple Linear Regression baseline?
3. Does XGBoost outperform a Random Forest reference model?
4. Does hyperparameter tuning improve the initial XGBoost model?
5. Which economic indicators are most important for prediction?
6. Does predictive accuracy differ across EU economies at different GDP-per-capita levels?
7. Which countries are hardest to predict, and are group-level differences broad patterns or driven by a few unusual economies?

---

## Data

### Source

All data comes from the **World Bank's World Development Indicators (WDI)** database:
- Main database: https://databank.worldbank.org/source/world-development-indicators

### Scope

- **Population**: 27 current EU member states
- **Period**: 1995–2025 (target: 1996–2024 for next-year prediction)
- **Panel size**: 810 country-year observations after preprocessing

### Core predictors

| Indicator | World Bank Code |
|---|---|
| GDP growth (annual %) | `NY.GDP.MKTP.KD.ZG` |
| GDP per capita (constant 2015 US$) | `NY.GDP.PCAP.KD` |
| Inflation, consumer prices (annual %) | `FP.CPI.TOTL.ZG` |
| Gross fixed capital formation (% of GDP) | `NE.GDI.FTOT.ZS` |
| Unemployment (% of labor force) | `SL.UEM.TOTL.ZS` |
| Exports (% of GDP) | `NE.EXP.GNFS.ZS` |
| Imports (% of GDP) | `NE.IMP.GNFS.ZS` |
| FDI, net inflows (% of GDP) | `BX.KLT.DINV.WD.GD.ZS` |
| Government consumption (% of GDP) | `NE.CON.GOVT.ZS` |
| Population growth (annual %) | `SP.POP.GROW` |

### Target

**Next-year GDP growth** — the value of `gdp_growth` one year ahead, computed within each country via `groupby("Country Code").shift(-1)`.

### Feature engineering

- **Two GDP growth lags** (lag-1, lag-2), computed within country.
- **Trade openness** = exports + imports (% of GDP).
- **GDP per capita groups** (tertiles of country mean) for post-model analysis only — not a feature.
- **Chronological train/test split**: train 1995–2019, test 2020–2024.

### How to access the data

The raw data is included in `data/raw/`. To re-download from source:
1. Go to https://databank.worldbank.org/source/world-development-indicators
2. Select all 27 EU countries
3. Select the indicators listed above
4. Set time range 1995–2025
5. Export as CSV with years as columns
---

## Methodology

### 1. Preprocessing (`src/preprocessing.py`)

- Load the reshaped World Bank panel.
- Create lagged GDP growth features within country.
- Create the next-year target.
- Drop rows without a target (2025).
- Assign GDP-per-capita tertiles for post-model analysis.
- Assign the chronological train/test split.
- Impute missing FDI (6 rows, all Luxembourg 1996–2001) using country-level median, fit on training rows only.

### 2. Feature expansion experiment (`src/preprocessing_extra.py`)

Five additional World Bank indicators were tested for inclusion. Four were rejected on data-availability grounds (22.9%–74.8% missing, structurally concentrated in early years). One — current account balance — was viable (5.1% missing) and improved test MAE by 3.2%. Documented in `08_feature_expansion_experiment.ipynb` but **not integrated** into the final model.

### 3. Models (`src/models.py`)

Five models, all wrapped in scikit-learn Pipelines (median imputation + standard scaling):

- **Naive baseline — mean**: predicts training mean for every test row.
- **Naive baseline — persistence**: predicts next year's growth = this year's growth.
- **Linear Regression**: OLS baseline.
- **Random Forest**: tree ensemble reference model.
- **XGBoost**: flagship model.

### 4. Hyperparameter tuning (`05_hyperparameter_tuning.ipynb`)

Three-stage tuning of XGBoost:

1. **Random Search** — 60 candidates, 5-fold `TimeSeriesSplit` CV.
2. **Grid Search** — 324 candidates around Random Search best.
3. **Manual tuning** — 4 hand-picked extreme configurations to probe the bias-variance boundary.

Test set never used for tuning decisions.

### 5. Evaluation (`src/evaluate.py`)

- **Primary metric**: MAE (interpretable in percentage points of GDP growth).
- **Secondary metrics**: RMSE, R².
- **Overfitting analysis**: train-test gap for all models.
- **Group analysis**: MAE by GDP-per-capita tertile.
- **Country analysis**: MAE per country.

---

## Key Findings

### Model comparison (test set, 2020–2024)

| Model | Test MAE | Test R² |
|---|---|---|
| **Naive: mean** | **2.490** | −0.050 |
| Linear Regression | 2.748 | −0.647 |
| XGBoost GridSearch (tuned) | 2.841 | **−0.332** |
| Random Forest | 3.131 | −0.523 |
| XGBoost baseline | 3.503 | −0.799 |
| Naive: persistence | 4.214 | −2.529 |

### Main findings

1. **No ML model beats the naive mean baseline on test MAE.** This is the project's central result and is consistent with the empirical forecasting literature (see Meese-Rogoff 1983; Beck et al. 2025).

2. **Hyperparameter tuning helped substantially.** XGBoost test MAE fell from 3.50 (baseline) to 2.84 (tuned GridSearch) — a 19% improvement. Hypothesis 3 supported.

3. **Tuning did not overcome the fundamental limitation.** All test R² remain negative; the model does not explain more variance than the test mean.

4. **Overfitting grows with model capacity.** Train-test MAE gap: Naive ≈ 0, Linear Regression 0.51, tuned XGBoost 1.69, Random Forest 2.17, untuned XGBoost 2.72.

5. **Feature importance is diffuse.** Even `gdp_growth` (the strongest feature) captures only 15% of model importance. Most structural indicators contribute little.

6. **Country-level differences dominate group-level differences.** MAE by country ranges from 1.3 (Finland) to 6.7 (Ireland) — a factor of 5×. GDP-per-capita group differences are much smaller and appear driven by individual countries (Ireland, Malta) rather than a broad development-related pattern.

7. **Feature expansion was investigated rigorously.** Five candidate indicators were tested; four rejected on coverage grounds; the single viable one (current account balance) produced a small improvement that did not change the project's overall conclusion.

---

## Repository Structure
```
ML Project/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                          # original World Bank downloads
│   └── processed/                    # cleaned, model-ready panels
├── src/                              # reusable Python modules
│   ├── config.py                     # paths, column names, constants, seed
│   ├── preprocessing.py              # main data pipeline
│   ├── preprocessing_extra.py        # feature expansion loader + merge
│   ├── split.py                      # train/test split + TimeSeriesSplit CV
│   ├── models.py                     # model builders (5 models)
│   ├── evaluate.py                   # metrics + comparison tables
│   └── plots.py                      # reusable plotting functions
└── notebooks/
    ├── 01_data_collection_and_cleaning.ipynb
    ├── 02_eda.ipynb
    ├── 03_feature_engineering.ipynb
    ├── 04_model_training.ipynb
    ├── 05_hyperparameter_tuning.ipynb
    ├── 06_evaluation_and_findings.ipynb
    ├── 07_data_collection_and_cleaning_extra.ipynb
    └── 08_feature_expansion_experiment.ipynb
```

## How to Reproduce

### Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
jupyter
```

Install with:
```bash
pip install -r requirements.txt
```

Run
Clone the repository.

Ensure raw data is in data/raw/.

Run the notebooks in order (01 → 08). All random seeds are set in src/config.py (RANDOM_STATE = SEED); results are deterministic.

Module imports
Notebooks use:
```python
import sys
from pathlib import Path
current = Path.cwd()
project_root = current.parent if current.name == "notebooks" else current
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from src import config, split, models, evaluate
```

## Technologies
Python 3.13

pandas / numpy — data handling

scikit-learn — pipelines, preprocessing, models, cross-validation

XGBoost — flagship gradient-boosted model

matplotlib / seaborn — visualisation

## Limitations
Small sample: 675 training rows is modest for ML.

Chronological test period (2020–2024) contains two exceptional shocks: the COVID-19 crash and the post-pandemic rebound. No model trained on pre-2020 data can anticipate these.

Panel dependence: observations within countries are not independent.

Feature importance is not causal. A high-importance feature is not a cause of GDP growth.

Group analysis uses only 45 test observations per group. Differences should be read as suggestive.

Data quality issues: World Bank indicators contain measurement issues (e.g., Ireland's 2015 GDP restatement of ~+25%, driven by a multinational accounting event).

Coverage constraints: many WDI indicators have structural gaps for EU-27, 1995–2025.

## Presentation
Slides: [text](https://docs.google.com/presentation/d/17DnXfqTFIbtQyj0RGZyEhsJ_j7WFl80fauf3-RYB-f8/edit?usp=drive_link)

## References
Meese, R., & Rogoff, K. (1983). Empirical exchange rate models of the seventies: Do they fit out-of-sample? Journal of International Economics.

Beck, Dovern & Vogl (2025). Mind the naive forecast! A rigorous evaluation of forecasting models for time series with low predictability. Applied Intelligence.

World Bank. World Development Indicators. https://databank.worldbank.org/source/world-development-indicators

## Authors
Ana Preto
Sevgi Ozdemir

## License
This project is submitted as coursework. Data is publicly available from the World Bank under their terms of use.

