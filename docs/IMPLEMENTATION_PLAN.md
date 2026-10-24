# 🏠 End-to-End House Price Predictor — Implementation Plan

> **Goal**: Build a production-ready house price prediction system from raw data to a deployed web app, learning core Data Science skills at each step.
>
> **Dataset**: Ames Housing Dataset (Kaggle) — 1,460 training rows, 81 features, target = `SalePrice`
>
> **Status Legend**: ✅ Done · 🔄 In Progress · ⬜ Pending

---

## Project Structure

```
End-to-End House Price Predictor/
├── data/
│   ├── raw/                    # Immutable source data — never modify
│   └── processed/              # Cleaned & feature-engineered data
├── notebooks/                  # One .ipynb per phase (exploration & learning)
├── src/                        # Reusable Python modules (called from notebooks)
├── models/                     # Serialized trained models (.pkl / .json)
├── plots/                      # Phase-wise saved charts
├── app/                        # Streamlit deployment app
└── docs/                       # This file and other documentation
```

---

## Phase Overview

| Phase | Name | Notebook | Status |
|-------|------|----------|--------|
| 1 | Exploratory Data Analysis | `01_phase1_eda.ipynb` | ✅ Done |
| 2 | Data Cleaning | `02_phase2_cleaning.ipynb` | ⬜ Pending |
| 3 | Feature Engineering | `03_phase3_features.ipynb` | ⬜ Pending |
| 4 | Baseline Modeling | `04_phase4_baseline.ipynb` | ⬜ Pending |
| 5 | Advanced Models & Tuning | `05_phase5_advanced.ipynb` | ⬜ Pending |
| 6 | Model Evaluation | `06_phase6_evaluation.ipynb` | ⬜ Pending |
| 7 | Deployment | `app/streamlit_app.py` | ⬜ Pending |

---

## Phase 1 — Exploratory Data Analysis ✅

**File**: `notebooks/01_phase1_eda.ipynb`  
**Script**: `phase1_eda.py`

### What was done
- Loaded and inspected `train.csv` (1,460 × 81) and `test.csv` (1,459 × 80)
- Identified 38 numerical and 43 categorical columns
- Analysed 19 columns with missing values (`PoolQC` at 99.5% worst)
- Confirmed `SalePrice` is right-skewed (skew = 1.88) → needs `log1p` transform
- Built correlation matrix → top predictor: `OverallQual` (r = 0.791)
- Detected 2 extreme outliers: `GrLivArea > 4000` with `SalePrice < $200k`

### Key Findings
| Finding | Value |
|---------|-------|
| Strongest predictor | `OverallQual` (r = 0.791) |
| SalePrice skewness | 1.88 (needs log-transform) |
| Features with missing data | 19 |
| Extreme outliers detected | 2 rows (Id: 524, 1299) |

### Plots Generated (`plots/phase1/`)
- `01_missing_values.png`
- `02_saleprice_distribution.png`
- `03_numerical_distributions.png`
- `04_correlation_heatmap.png`
- `05_categorical_vs_price.png`
- `06_outlier_inspection.png`
- `07_quality_vs_price.png`

---

## Phase 2 — Data Cleaning ⬜

**File**: `notebooks/02_phase2_cleaning.ipynb`  
**Output**: `data/processed/train_cleaned.csv`, `data/processed/test_cleaned.csv`

### Tasks
- [ ] **Remove outliers**: Drop rows where `GrLivArea > 4000` AND `SalePrice < 200000`
- [ ] **Log-transform target**: `train['SalePrice'] = np.log1p(train['SalePrice'])`
- [ ] **Impute missing values**:
  - `PoolQC`, `MiscFeature`, `Alley`, `Fence`, `FireplaceQu`, `Garage*`, `Bsmt*` → fill with `"None"` (absence of feature)
  - `LotFrontage` → fill with median grouped by `Neighborhood`
  - `MasVnrType`, `MasVnrArea` → fill with `"None"` / `0`
  - `GarageYrBlt` → fill with `0`
  - `Electrical` (1 missing) → fill with mode
- [ ] **Fix data types**: Convert year columns and ordinal codes to integers
- [ ] **Handle rare categories**: Group very infrequent category values into `"Other"`
- [ ] **Save cleaned data** to `data/processed/`

### Concepts You'll Learn
- Domain-aware vs. statistical imputation strategies
- Why `SimpleImputer` from scikit-learn exists (and when to use it vs. manual logic)
- The difference between `fillna` strategies (mean, median, mode, constant)

---

## Phase 3 — Feature Engineering ⬜

**File**: `notebooks/03_phase3_features.ipynb`  
**Output**: `data/processed/train_features.csv`, `data/processed/test_features.csv`

### Tasks
- [ ] **Create new features**:
  - `TotalSF` = `TotalBsmtSF` + `1stFlrSF` + `2ndFlrSF`
  - `TotalBath` = `FullBath` + `0.5 × HalfBath` + `BsmtFullBath` + `0.5 × BsmtHalfBath`
  - `HouseAge` = `YrSold` - `YearBuilt`
  - `RemodAge` = `YrSold` - `YearRemodAdd`
  - `IsRemodeled` = 1 if `YearRemodAdd != YearBuilt` else 0
  - `HasPool`, `HasGarage`, `Has2ndFloor`, `HasFireplace` (binary flags)
- [ ] **Encode categorical features**:
  - Ordinal columns (e.g., `ExterQual`, `BsmtQual`) → map to integer scale (0–4)
  - Nominal columns → One-Hot Encode with `pd.get_dummies` (or `OneHotEncoder`)
- [ ] **Fix skewed numerical features**: Apply `np.log1p` to features with skewness > 0.75
- [ ] **Drop low-value / leakage-prone columns**: `Id`, `MoSold`, `YrSold` (post-encoding)
- [ ] **Align train and test columns** (test may be missing some dummies)
- [ ] **Save feature set** to `data/processed/`

### Concepts You'll Learn
- Feature creation from domain knowledge
- Ordinal vs nominal encoding and when to use each
- Why skewness in features (not just target) matters for linear models
- The `get_dummies` alignment trick for train/test parity

---

## Phase 4 — Baseline Modeling ⬜

**File**: `notebooks/04_phase4_baseline.ipynb`  
**Output**: models saved in `models/`

### Tasks
- [ ] **Split data**: `train_test_split` (80/20), stratified on price bin
- [ ] **Build baseline models**:
  1. `LinearRegression` — simplest possible model
  2. `Ridge` (L2 regularization) — prevents overfitting on correlated features
  3. `Lasso` (L1 regularization) — auto feature selection via zeroed weights
  4. `DecisionTreeRegressor` — non-linear, no scaling needed
- [ ] **Evaluate each model**:
  - RMSE (Root Mean Squared Error) on log-prices
  - R² score (Coefficient of Determination)
  - 5-Fold Cross-Validation mean and std
- [ ] **Compare results in a summary table**
- [ ] **Save best baseline model** with `joblib.dump(model, 'models/baseline_ridge.pkl')`

### Concepts You'll Learn
- The bias-variance tradeoff (underfitting vs overfitting)
- Why scaling matters for Ridge/Lasso but not for trees
- Cross-validation vs simple train/test split
- RMSE on log-space vs original price space

---

## Phase 5 — Advanced Models & Hyperparameter Tuning ⬜

**File**: `notebooks/05_phase5_advanced.ipynb`

### Tasks
- [ ] **Train ensemble models**:
  1. `RandomForestRegressor` — bagging (many trees in parallel)
  2. `GradientBoostingRegressor` — boosting (sequential correction)
  3. `XGBRegressor` (XGBoost) — industry-standard gradient boosting
  4. `LGBMRegressor` (LightGBM) — faster alternative to XGBoost
- [ ] **Hyperparameter tuning**:
  - `GridSearchCV` for Ridge alpha
  - `RandomizedSearchCV` for XGBoost / LightGBM
- [ ] **Stacking / Blending** *(stretch goal)*:
  - Average predictions from Ridge + XGBoost + LightGBM
- [ ] **Feature importance analysis**:
  - Plot top 20 features from XGBoost
  - SHAP values for explainability *(stretch goal)*
- [ ] **Save best model** to `models/best_model.pkl`

### Concepts You'll Learn
- Bagging vs Boosting (Random Forest vs XGBoost)
- Hyperparameter tuning strategies: Grid, Random, Bayesian
- Ensemble methods and model blending
- Feature importance and explainability

---

## Phase 6 — Model Evaluation & Submission ⬜

**File**: `notebooks/06_phase6_evaluation.ipynb`

### Tasks
- [ ] **Full residual analysis**: Plot actual vs predicted, residual distribution
- [ ] **Learning curves**: Diagnose if model is under/overfitting
- [ ] **Error analysis**: What kinds of houses does the model struggle with?
- [ ] **Generate Kaggle submission**:
  - Load `data/raw/test.csv` → run full pipeline → inverse `expm1` → write `submission.csv`
- [ ] **Submit to Kaggle** and note the leaderboard RMSE score

### Concepts You'll Learn
- Diagnostic plots for regression models
- The pipeline concept: `sklearn.pipeline.Pipeline` for repeatable inference
- The importance of `np.expm1` when using log-transformed targets

---

## Phase 7 — Streamlit Web App Deployment ⬜

**File**: `app/streamlit_app.py`

### Tasks
- [ ] **Build interactive UI** with sliders/dropdowns for key house features:
  - `OverallQual`, `GrLivArea`, `TotalBsmtSF`, `GarageArea`, `Neighborhood`, `YearBuilt`, etc.
- [ ] **Load saved model** (`models/best_model.pkl`) at app startup
- [ ] **Prediction pipeline**:
  - Input → feature engineer → apply encoders → model.predict → `np.expm1` → display price
- [ ] **Visualization**: Show which input features are most influential (feature importance bar)
- [ ] **Run locally**: `streamlit run app/streamlit_app.py`
- [ ] *(Optional)* Deploy to Streamlit Community Cloud (free hosting)

### Concepts You'll Learn
- How to serialize/deserialize ML pipelines
- Building real UIs for ML models without web dev experience
- The full prediction loop from user input to model output

---

## Skills & Libraries Progression

```
Phase 1  → pandas, numpy, matplotlib, seaborn
Phase 2  → pandas (advanced), sklearn.impute
Phase 3  → sklearn.preprocessing, feature engineering
Phase 4  → sklearn.linear_model, sklearn.tree, sklearn.model_selection
Phase 5  → sklearn.ensemble, xgboost, lightgbm, sklearn.pipeline
Phase 6  → sklearn.metrics, joblib, full pipeline inference
Phase 7  → streamlit, joblib (deployment)
```

---

## Success Metrics

| Metric | Baseline Target | Stretch Target |
|--------|----------------|----------------|
| Kaggle RMSE (log-price) | < 0.15 | < 0.12 |
| R² Score (test split) | > 0.88 | > 0.92 |
| Model deployed & live | Locally running | Streamlit Cloud |
| Code quality | Clean notebooks | Modular `src/` functions |

---

*Last updated: October 2026*
