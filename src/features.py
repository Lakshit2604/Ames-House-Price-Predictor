"""
src/features.py
---------------
Reusable feature engineering functions for the House Price Predictor project.
Imported by: notebooks/03_phase3_features.ipynb

Functions:
    create_domain_features      — build interaction, total area, age, and binary flags
    encode_ordinal_features     — map ordered category strings to integer scales
    log_transform_skewed        — apply log1p to right-skewed continuous numerical features
    one_hot_encode_and_align    — dummy encode nominal variables and align train/test cols
    feature_engineering_report  — print summary of engineered features & correlations
"""

import numpy as np
import pandas as pd


# ── 1. Create Domain Features ───────────────────────────────────────────
def create_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer domain-specific features based on real estate heuristics:
    - TotalSF: Total square footage across basement, 1st, and 2nd floors
    - TotalBath: Weighted bathroom count (full + 0.5 * half)
    - HouseAge: Age of house at time of sale
    - RemodAge: Years since last remodel at time of sale
    - IsRemodeled: Binary flag indicating if remodel occurred
    - Binary presence flags: HasPool, HasGarage, Has2ndFloor, HasFireplace, HasBsmt
    - Interaction features: OverallScore (Qual * Cond), TotalPorchSF
    """
    df = df.copy()

    # Total area & bath combinations
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBath"] = (
        df["FullBath"]
        + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"]
        + 0.5 * df["BsmtHalfBath"]
    )

    # Age features (clamped at 0 to guard against edge-case negative values)
    yr_sold = df["YrSold"].astype(int)
    yr_built = df["YearBuilt"].astype(int)
    yr_remod = df["YearRemodAdd"].astype(int)

    df["HouseAge"] = np.maximum(0, yr_sold - yr_built)
    df["RemodAge"] = np.maximum(0, yr_sold - yr_remod)
    df["IsRemodeled"] = (yr_remod != yr_built).astype(int)

    # Binary presence indicators
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)

    # Interactions & composite porch area
    df["OverallScore"] = df["OverallQual"] * df["OverallCond"]
    df["TotalPorchSF"] = (
        df["OpenPorchSF"]
        + df["EnclosedPorch"]
        + df["3SsnPorch"]
        + df["ScreenPorch"]
    )

    print("[create_domain_features] Created 12 domain features.")
    return df


# ── 2. Ordinal Feature Encoding ─────────────────────────────────────────
def encode_ordinal_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Map ordered categorical ratings to numeric scales (e.g. None=0, Po=1, ..., Ex=5).
    Preserves ordinal hierarchy and avoids explosive one-hot dimensionality.
    """
    train = train.copy()
    test  = test.copy()

    # Standard quality/condition mapping (5-point scale)
    qual_map = {
        "None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5, "Other": 2
    }
    qual_cols = [
        "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
        "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"
    ]
    for col in qual_cols:
        if col in train.columns:
            train[col] = train[col].map(qual_map).fillna(0).astype(int)
        if col in test.columns:
            test[col] = test[col].map(qual_map).fillna(0).astype(int)

    # Specific ordinal maps
    specific_maps = {
        "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4, "Other": 1},
        "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6, "Other": 2},
        "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6, "Other": 2},
        "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3, "Other": 1},
        "Fence":        {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4, "Other": 2},
        "Functional":   {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7, "Other": 5},
        "LandSlope":    {"Sev": 1, "Mod": 2, "Gtl": 3, "Other": 2},
        "LotShape":     {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4, "Other": 2},
        "PavedDrive":   {"N": 0, "P": 1, "Y": 2, "Other": 1},
    }

    for col, mapping in specific_maps.items():
        if col in train.columns:
            train[col] = train[col].map(mapping).fillna(0).astype(int)
        if col in test.columns:
            test[col] = test[col].map(mapping).fillna(0).astype(int)

    total_encoded = len(qual_cols) + len(specific_maps)
    print(f"[encode_ordinal_features] Mapped {total_encoded} ordinal features to integers.")
    return train, test


# ── 3. Log-Transform Skewed Numerical Features ─────────────────────────
def log_transform_skewed(
    train: pd.DataFrame,
    test: pd.DataFrame,
    threshold: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Identify continuous numeric features with absolute skewness > threshold on train,
    and apply np.log1p() to both train and test.
    Excludes identifiers, targets, binary flags, year timestamps, categorical codes, and ordinal scales.
    """
    train = train.copy()
    test  = test.copy()

    # Features that should never be log1p transformed
    protected = {
        "Id", "SalePrice", "MSSubClass", "IsRemodeled", "HasPool", "HasGarage",
        "Has2ndFloor", "HasFireplace", "HasBsmt", "OverallQual", "OverallCond",
        "OverallScore", "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath",
        "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageCars",
        "YearBuilt", "YearRemodAdd", "GarageYrBlt", "YrSold", "MoSold",
        "HouseAge", "RemodAge", "ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
        "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond",
        "PoolQC", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "GarageFinish",
        "Fence", "Functional", "LandSlope", "LotShape", "PavedDrive",
    }

    candidates = [
        c for c in train.select_dtypes(include=[np.number]).columns
        if c not in protected
    ]

    skews = train[candidates].skew()
    high_skew_cols = skews[skews.abs() > threshold].index.tolist()

    for col in high_skew_cols:
        train[col] = np.log1p(train[col])
        if col in test.columns:
            test[col] = np.log1p(test[col])

    print(f"[log_transform_skewed] Log1p transformed {len(high_skew_cols)} skewed features (skew > {threshold}).")
    return train, test, high_skew_cols


# ── 4. One-Hot Encoding and Column Alignment ────────────────────────────
def one_hot_encode_and_align(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-Hot encode all nominal categorical columns using pd.get_dummies(drop_first=True).
    Aligns train and test columns so both datasets have an identical feature space.
    Preserves 'Id' and 'SalePrice' columns.
    """
    train = train.copy()
    test  = test.copy()

    # Ensure pseudo-numeric categoricals are string before dummification
    for col in ["MSSubClass", "MoSold"]:
        if col in train.columns:
            train[col] = train[col].astype(str)
        if col in test.columns:
            test[col] = test[col].astype(str)

    # Save target and IDs
    y_train = train["SalePrice"]
    train_id = train["Id"]
    test_id  = test["Id"]

    # Drop target and ID for dummification
    X_tr = train.drop(columns=["Id", "SalePrice"])
    X_te = test.drop(columns=["Id"])

    # One-hot encode nominal columns
    X_tr_enc = pd.get_dummies(X_tr, drop_first=True, dtype=int)
    X_te_enc = pd.get_dummies(X_te, drop_first=True, dtype=int)

    # Align columns: test matches train columns exactly (missing = 0)
    X_tr_aligned, X_te_aligned = X_tr_enc.align(
        X_te_enc, join="left", axis=1, fill_value=0
    )

    # Reattach Id and SalePrice
    train_final = pd.concat([train_id, X_tr_aligned, y_train], axis=1)
    test_final  = pd.concat([test_id, X_te_aligned], axis=1)

    print(f"[one_hot_encode_and_align] Train: {train_final.shape}, Test: {test_final.shape} ({X_tr_aligned.shape[1]} features aligned).")
    return train_final, test_final


# ── 5. Feature Engineering Report ───────────────────────────────────────
def feature_engineering_report(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Print an executive summary of the engineered feature datasets."""
    sep = "─" * 60
    print(f"\n{sep}")
    print("  FEATURE ENGINEERING SUMMARY REPORT")
    print(sep)
    print(f"  Train shape : {train.shape[0]:,} rows × {train.shape[1]} columns")
    print(f"  Test  shape : {test.shape[0]:,} rows × {test.shape[1]} columns")

    train_nulls = train.isnull().sum().sum()
    test_nulls  = test.isnull().sum().sum()
    print(f"  Null count  : train = {train_nulls}, test = {test_nulls}")

    features = [c for c in train.columns if c not in ["Id", "SalePrice"]]
    print(f"  Total model features : {len(features)}")

    if "SalePrice" in train.columns:
        corrs = train[features].corrwith(train["SalePrice"]).abs().sort_values(ascending=False)
        print(f"\n  Top 10 features by absolute correlation with log(SalePrice):")
        for i, (feat, r) in enumerate(corrs.head(10).items(), 1):
            sign = "+" if train[feat].corr(train["SalePrice"]) >= 0 else "-"
            print(f"    {i:2d}. {feat:<22} : {sign}{r:.4f}")
    print(sep)
