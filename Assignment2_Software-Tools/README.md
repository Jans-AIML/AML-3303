# Predicting Airbnb Listing Prices with MLflow and AWS S3

**AML-3303 — Assignment 2 | StayWise Data Science Team**

> **Objective:** Build a clean, reproducible machine learning pipeline that predicts the optimal nightly price for new Airbnb listings in New York City, using MLflow for full experiment tracking and AWS S3 for cloud data retrieval.
>
> Two iterations of this pipeline were produced. **Experiment 1** is the original baseline notebook. **Experiment 2** introduces carefully designed preprocessing, targeted outlier imputation, class resampling, a leaner 8-feature set, and the addition of XGBRegressor — yielding a substantially better Random Forest model.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Setup & Execution](#2-setup--execution)
3. [Repository Structure](#3-repository-structure)
4. [Pipeline Workflow](#4-pipeline-workflow)
5. [MLflow Experiment Tracking](#5-mlflow-experiment-tracking)
6. [Key Results & Insights](#6-key-results--insights)
7. [Experiment Comparison](#7-experiment-comparison)
8. [Plots & Visualisations](#8-plots--visualisations)

---

## 1. Project Overview

### Business Problem

Airbnb hosts often struggle to price new listings competitively. Under-pricing leaves revenue on the table; over-pricing reduces bookings. This project builds a **nightly price prediction model** trained on 48,000+ historical NYC listings to give data-driven pricing recommendations.

### Objectives

| # | Objective |
|---|-----------|
| 1 | Retrieve the raw dataset from a public **AWS S3** bucket using anonymous boto3 access. |
| 2 | Perform **Exploratory Data Analysis (EDA)** and end-to-end preprocessing. |
| 3 | Train and compare **regression models** with 5-fold cross-validation across two experiment iterations. |
| 4 | Track all experiments, metrics, parameters, and artefacts using **MLflow**. |
| 5 | Run a **27-run hyperparameter tuning grid** on the best model (Random Forest). |
| 6 | Register the optimal model in the **MLflow Model Registry** and promote it to production. |

### Dataset

| Property | Value |
|----------|-------|
| Source | `s3://staywise1/airbnb/raw_data/AB_NYC_2019.csv` (public bucket). |
| Rows | ~48,895 listings. |
| Target variable | **Exp. 1:** `log_price` — `np.log1p(price)`. **Exp. 2:** `log_total_cost` — `np.log1p(price × minimum_nights)`. |
| Geography | New York City, 5 boroughs, 221 neighbourhoods. |

---

## 2. Setup & Execution

### Prerequisites

- Python **3.10+**
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Jans-AIML/AML-3303.git
cd AML-3303/Assignment2_Software-Tools

# 2. (Optional) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows PowerShell

# 3. Install all dependencies
pip install -r requirements.txt
```

### Running the Notebook

```bash
# Experiment 1 — baseline
jupyter notebook "notebook/Airbnb_Listing Prices_MLflow_AWS-S3.ipynb"

# Experiment 2 — improved preprocessing + XGBRegressor
jupyter notebook "notebook/Airbnb_Listing Prices_MLflow_AWS-S3_v2.ipynb"

# Or open directly in VS Code
code "notebook/Airbnb_Listing Prices_MLflow_AWS-S3_v2.ipynb"
```

Run the cells **in order** (top to bottom). Each step builds on the previous one.

**Experiment 1** (`Airbnb_Listing Prices_MLflow_AWS-S3.ipynb`):

| Cells | Step | Description |
|-------|------|-------------|
| 1–3 | Setup | Install packages, configure logging. |
| 4–8 | Step 1 | Load data from public S3 bucket (no credentials needed). |
| 9–17 | Step 2 | EDA, cleaning, feature engineering, encoding. |
| 18–24 | Step 3 | Train & compare 5 models, leaderboard, visualisations. |
| 25–28 | Step 4 | Log all runs to MLflow, compare from registry. |
| 29–32 | Step 4b | RF hyperparameter tuning (27-run grid), best config registration. |

**Experiment 2** (`Airbnb_Listing Prices_MLflow_AWS-S3_v2.ipynb`):

| Cells | Step | Description |
|-------|------|-------------|
| 1–3 | Setup | Install packages (incl. xgboost), configure logging. |
| 4–8 | Step 1 | Load data from public S3 bucket (no credentials needed). |
| 9–17 | Step 2 | Targeted outlier imputation, class resampling, 8-feature engineering, LabelEncoding. |
| 18–25 | Step 3 | Train & compare 5 models incl. XGBRegressor, leaderboard, visualisations. |
| 26–29 | Step 4 | Log all runs to MLflow, register best model, Actual vs Predicted plot. |

### Launching the MLflow UI

```bash
# From the Assignment2_Software-Tools folder
mlflow ui

# Then open in browser:
# http://127.0.0.1:5000
```

> **No AWS credentials are required.** The S3 bucket is publicly readable and all requests are sent unsigned via `Config(signature_version=UNSIGNED)`.

---

## 3. Repository Structure

```
AML-3303/
├── .gitignore                          # Excludes datasets, MLflow runs, env folders
├── Assignment2_Software-Tools/
│   ├── README.md                       # ← This file
│   ├── requirements.txt                # Pinned Python dependencies
│   ├── data/
│   │   └── Ab Nyc2019 Data Dictionary.docx   # Column descriptions
│   └── notebook/
│       ├── Airbnb_Listing Prices_MLflow_AWS-S3.ipynb    # Experiment 1 — baseline
│       └── Airbnb_Listing Prices_MLflow_AWS-S3_v2.ipynb # Experiment 2 — improved
├── Assessment1/                        # Separate assessment (ConnectWave churn)
├── Week1/ … Week4/                     # Weekly exercises
└── LICENSE
```

> **Note:** Raw CSV data (`AB_NYC_2019.csv`) and MLflow tracking artefacts (`mlruns/`) are excluded from version control via `.gitignore`. They are generated locally when the notebook is executed.

---

## 4. Pipeline Workflow

Both experiments share the same four-step skeleton. The key differences lie in Step 2 (preprocessing strategy) and the model set in Step 3.

### Experiment 1 — Baseline

```
S3 (public bucket)
        │
        ▼
┌───────────────┐
│   Step 1      │  boto3 unsigned → stream CSV → pd.DataFrame
│  Data Load    │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   Step 2      │  Drop IDs · Filter price==0 · Impute reviews_per_month
│  EDA &        │  Cap outliers (p99) · Engineer features · Encode categoricals
│  Preprocessing│  → df_clean  (48k rows × ~16 features)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   Step 3      │  train_test_split (80/20) · StandardScaler (train-only)
│  Model        │  5-fold CV + hold-out evaluation for:
│  Comparison   │  LinearRegression | Ridge | Lasso | RandomForest | GradientBoosting
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   Step 4      │  mlflow.set_experiment("airbnb-price-prediction")
│  MLflow       │  Log params + metrics + model artefact + preprocessing artefacts
│  Tracking     │  Query runs · Compare leaderboard · Register best model
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   Step 4b     │  ParameterGrid: n_estimators × max_depth × min_samples_leaf
│  RF Tuning    │  27 runs → "rf-hyperparameter-tuning" experiment
│  Experiment   │  Heatmaps + line plots · Register best tuned config → production
└───────────────┘
```

### Experiment 2 — Improved Preprocessing

```
S3 (public bucket)
        │
        ▼
┌───────────────┐
│   Step 1      │  boto3 unsigned → stream CSV → pd.DataFrame
│  Data Load    │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   Step 2      │  Drop IDs · Impute reviews_per_month (NaN → 0)
│  EDA &        │  Targeted outlier imputation (price, minimum_nights, availability_365)
│  Preprocessing│  using borough × room_type category medians (no rows removed)
│               │  Oversample minority classes (neighbourhood_group & room_type)
│               │  Engineer 3 features · LabelEncode 2 categoricals
│               │  → df_clean  (resampled × 8 features)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   Step 3      │  train_test_split (80/20) · StandardScaler (train-only)
│  Model        │  5-fold CV + hold-out evaluation for:
│  Comparison   │  LinearRegression | Lasso | RandomForest | GradientBoosting | XGBRegressor
└───────┬───────┘
        │
        ▼
┌───────────────┐
│   Step 4      │  mlflow.set_experiment("airbnb-price-prediction")
│  MLflow       │  Log params + metrics + model artefact + encoding_metadata
│  Tracking     │  Query runs · Register best model · Actual vs Predicted plot
└───────────────┘
```

### Feature Engineering Comparison

| Feature | Experiment 1 | Experiment 2 |
|---------|:---:|:---:|
| `log_price` (target) | ✓ | — |
| `log_total_cost` (target = log1p(price × min_nights)) | — | ✓ |
| `neighbourhood_group_encoded` | — | ✓ |
| `latitude` / `longitude` | — | ✓ |
| `room_type_encoded` | — | ✓ |
| `number_of_reviews` | — | ✓ |
| `reviews_per_month` | — | ✓ |
| `calculated_host_listings_count` | — | ✓ |
| `availability_yearly` (availability_365 / 365) | — | ✓ |
| `days_since_last_review` | ✓ | — |
| `has_reviews` | ✓ | — |
| `reviews_per_min_night` | ✓ | — |
| `neighbourhood_encoded` (target-mean, 221 values) | ✓ | — |
| `room_type_*` (one-hot) | ✓ | — |
| **Total features** | **~16** | **8** |

---

## 5. MLflow Experiment Tracking

Both notebooks log to the same MLflow experiment name (`airbnb-price-prediction`), so all runs are queryable and comparable in a single view.

### What is logged per run

| Category | Details |
|----------|---------|
| **Parameters** | All sklearn/XGBoost hyperparameters, `n_features`, `use_scaled_input`, `train_rows`, `test_rows`, `target` |
| **Metrics** | `cv_rmse_mean`, `cv_rmse_std`, `cv_r2_mean`, `test_rmse_log`, `test_mae_log`, `test_r2`, `test_rmse_usd`, `train_time_s` |
| **Model artefact** | Fitted estimator via `mlflow.sklearn.log_model()` with input schema example |
| **Preprocessing** | `scaler.pkl` — Exp. 1: `neighbourhood_price_map.json`; Exp. 2: `encoding_metadata.json`; `feature_config.json` |
| **Plots** | Feature importance charts for all tree-based models |

### Best model per experiment

| Experiment | Best Model | Test R² | Test RMSE ($) | Registered as |
|------------|-----------|:-------:|:-------------:|---------------|
| Exp. 1 — baseline | Random Forest | 0.6216 | $93 | `airbnb-price-predictor` v1 |
| Exp. 2 — improved | Random Forest | 0.9084 | $139 | `airbnb-price-predictor` v2 (production) |

> Note: RMSE ($) values are not directly comparable across experiments because Exp. 1 predicts `log_price` (nightly rate) while Exp. 2 predicts `log_total_cost` (price × minimum_nights). Exp. 2's higher RMSE in dollars reflects the larger absolute scale of the total booking cost target.

### Experiment: `rf-hyperparameter-tuning` (Experiment 1 only)

A 3×3×3 grid search over Random Forest hyperparameters — **27 runs** logged to a dedicated experiment.

| Hyperparameter | Values explored |
|----------------|----------------|
| `n_estimators` | 100, 200, 300 |
| `max_depth` | 10, 20, unlimited |
| `min_samples_leaf` | 2, 4, 8 |

Key finding: `min_samples_leaf=2` with `max_depth=unlimited` consistently yielded the best R², and `n_estimators` beyond 200 produced diminishing returns. The best tuned config is registered as a dedicated version in the MLflow Model Registry.

### MLflow Model Registry: `airbnb-price-predictor`

| Version | Source experiment | Algorithm | Test R² | RMSE ($) | Status |
|---------|------------------|-----------|:-------:|:--------:|--------|
| v1 | `airbnb-price-prediction` (Exp. 1) | Random Forest (baseline) | 0.6216 | $93 | archived |
| v2 | `rf-hyperparameter-tuning` (Exp. 1) | Random Forest (tuned) | > 0.6216 | < $93 | archived |
| v3 | `airbnb-price-prediction` (Exp. 2) | Random Forest (improved) | 0.9084 | $139 | **production** |

---

## 6. Key Results & Insights

### Model Comparison — Experiment 1 (Baseline)

Target: `log_price` = `np.log1p(nightly price)`. Dataset: ~48 k rows, ~16 features.

| Model | CV RMSE (log) | Test R² | Test RMSE ($) |
|-------|:---:|:---:|:---:|
| Linear Regression | ~0.44 | 0.5635 | $100 |
| Ridge (α=1.0) | ~0.44 | 0.5635 | $100 |
| Lasso (α=0.01) | ~0.45 | 0.5613 | $101 |
| Gradient Boosting | ~0.41 | 0.6178 | $95 |
| **Random Forest** | **~0.41** | **0.6216** | **$93** |

### Model Comparison — Experiment 2 (Improved)

Target: `log_total_cost` = `np.log1p(price × minimum_nights)`. Dataset: resampled, 8 features.

| Model | CV RMSE (log) | Test R² | Test RMSE ($) |
|-------|:---:|:---:|:---:|
| Lasso (α=0.01) | ~0.73 | 0.4885 | $236 |
| Linear Regression | ~0.72 | 0.4893 | $235 |
| Gradient Boosting | ~0.49 | 0.7556 | $189 |
| XGBRegressor | ~0.44 | 0.8172 | $176 |
| **Random Forest** | **~0.32** | **0.9084** | **$139** |

### Key Observations

**1. Log-transformation is essential**
The raw `price` column is heavily right-skewed (max $10,000). Applying `np.log1p()` produces a near-normal target distribution, which benefits all regression models. Experiment 2 goes further by predicting `log_total_cost` = `log1p(price × minimum_nights)`, capturing the guest's true booking expenditure rather than the nightly rate alone.

**2. Targeted imputation beats row-removal**
Experiment 1 caps outliers at the 99th percentile, discarding information. Experiment 2 replaces out-of-range values in `price`, `minimum_nights`, and `availability_365` with borough × room_type category medians, preserving every row and producing a cleaner signal for all models.

**3. Class resampling improves minority-group predictions**
The NYC dataset is heavily imbalanced (Manhattan ~21 k rows vs Staten Island ~373). Experiment 2 oversamples underrepresented boroughs and room types to the majority class count, giving the model equal exposure to all combinations during training.

**4. 8 carefully chosen features outperform ~16 engineered ones**
Experiment 2 drops high-cardinality and redundant columns, retaining only the 8 features most correlated with total booking cost. The leaner feature set reduces noise and improves the Random Forest's generalisation substantially.

**5. Ensemble models dominate linear ones in both experiments**
Random Forest and Gradient Boosting consistently outperform Linear Regression and Lasso on Test R² and RMSE ($), confirming strong non-linear relationships between listing attributes and price.

**6. `min_samples_leaf` is the most impactful RF tuning knob (Experiment 1)**
The 27-run hyperparameter heatmaps show that smaller leaf sizes consistently improve R². `max_depth=unlimited` combined with low `min_samples_leaf` tends to produce the best results; increasing `n_estimators` beyond 200 gives diminishing returns.

**7. SDLC principles throughout**
All functions carry type hints and docstrings. `logging` replaces `print` statements. Constants (`S3_BUCKET`, `TARGET`, `FEATURE_COLS`) are defined once at the top of each step. No secrets are hardcoded — S3 uses unsigned public access.

---

## 7. Experiment Comparison

| Aspect | Experiment 1 | Experiment 2 |
|--------|-------------|-------------|
| **Notebook** | `…_MLflow_AWS-S3.ipynb` | `…_MLflow_AWS-S3_v2.ipynb` |
| **Target variable** | `log_price` | `log_total_cost` |
| **Outlier handling** | p99 cap (row-preserving) | Category-based median imputation |
| **Class imbalance** | Not addressed | Oversampling (neighbourhood_group & room_type) |
| **Feature count** | ~16 | **8** |
| **Encoding** | Target-mean + one-hot | LabelEncoder (2 columns) |
| **Models** | LR, Ridge, Lasso, RF, GBM | LR, Lasso, RF, GBM, **XGBRegressor** |
| **RF tuning** | 27-run grid (Step 4b) | Not included |
| **Final plot** | Residuals vs Fitted | Actual vs Predicted Total Cost ($) |

---

## 8. Plots & Visualisations

**Experiment 1**

| Plot | Cell | Description |
|------|:----:|-------------|
| Price distribution (raw vs log1p) | 11 | Shows the skewness problem and why log-transform is used. |
| Missing value heatmap | 11 | Reveals ~20% missing rate in `reviews_per_month` and `last_review`. |
| Model comparison bar charts | 23 | R², RMSE (log), RMSE ($) for all 5 models; best highlighted in orange. |
| CV RMSE error-bar chart | 23 | 5-fold stability — ensemble models show lower variance. |
| Residuals vs fitted + distribution | 23 | Validates near-zero mean error and near-normality of residuals. |
| Feature importance (RF & GBM) | 24 | `neighbourhood_encoded` dominates; `room_type_Private room` is second. |
| MLflow run comparison charts | 27 | Reproduces model comparison from live MLflow data. |
| RF tuning — all 27 configs | 31 | Sorted bar chart, colour-coded by `max_depth`. |
| RF tuning — R² heatmaps (3×) | 31 | Pairwise HP interaction: `leaf×depth`, `n×leaf`, `n×depth`. |
| RF tuning — line plots (3×) | 31 | Effect of each HP on mean R². |

**Experiment 2**

| Plot | Cell | Description |
|------|:----:|-------------|
| Outlier % barplots | 11 | Share of outlier values in price, min_nights, availability_365. |
| Class imbalance barplots | 11 | Borough and room_type counts before resampling. |
| Resampling before/after (2×2) | 17 | neighbourhood_group and room_type counts before and after oversampling. |
| Model comparison bar charts | 23 | R², RMSE (log), RMSE ($) for all 5 models incl. XGBRegressor. |
| CV RMSE error-bar chart | 23 | 5-fold stability across the improved feature set. |
| Residuals vs fitted + distribution | 23 | Best-model residual diagnostics. |
| Feature importance (RF, GBM, XGB) | 24 | Top-8 feature importances for all tree-based models. |
| MLflow run comparison charts | 27 | R² and RMSE ($) bars from live MLflow experiment. |
| Actual vs Predicted Total Cost | 29 | Green scatter + red perfect-prediction diagonal in USD space. |

---

*Generated as part of AML-3303 Assignment 2 — April 2026.*
