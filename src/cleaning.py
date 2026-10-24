"""
src/cleaning.py
---------------
Reusable data cleaning functions for the House Price Predictor project.
Imported by: notebooks/02_phase2_cleaning.ipynb

Functions:
    remove_outliers         — drop extreme GrLivArea/SalePrice rows
    log_transform_target    — apply log1p to SalePrice
    impute_missing          — domain-aware missing value imputation
    fix_rare_categories     — consolidate low-frequency category labels
    fix_dtypes              — cast pseudo-numeric columns to string
    cleaning_report         — print a summary of the cleaned dataset
"""

import numpy as np
import pandas as pd


# ── 1. Outlier Removal ──────────────────────────────────────────────────
def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove homes with extremely large living area but suspiciously low prices.
    These are partial/agricultural sales flagged by the dataset author.

    Rule: GrLivArea > 4000 AND SalePrice < 200,000
    """
    mask = ~((df["GrLivArea"] > 4000) & (df["SalePrice"] < 200_000))
    removed = (~mask).sum()
    print(f"[remove_outliers] Removed {removed} row(s).")
    return df[mask].reset_index(drop=True)


# ── 2. Log-Transform Target ─────────────────────────────────────────────
def log_transform_target(df: pd.DataFrame, col: str = "SalePrice") -> pd.DataFrame:
    """
    Apply log1p to the target column in-place.
    Use np.expm1() to reverse the transform when generating predictions.
    """
    df = df.copy()
    df[col] = np.log1p(df[col])
    print(f"[log_transform_target] '{col}' transformed. New skew: {df[col].skew():.3f}")
    return df


# ── 3. Missing Value Imputation ─────────────────────────────────────────
def impute_missing(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Domain-aware imputation derived from data_description.txt.
    Applies the same transformations to both train and test to prevent leakage.

    Returns:
        (train_imputed, test_imputed)
    """
    train = train.copy()
    test  = test.copy()

    # Columns where NaN means the feature simply doesn't exist
    none_cols = [
        "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
        "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
        "MasVnrType",
    ]
    for col in none_cols:
        for df in (train, test):
            if col in df.columns:
                df[col] = df[col].fillna("None")

    # Numeric counterparts — if no garage/basement, area/count = 0
    zero_cols = [
        "GarageYrBlt", "GarageCars", "GarageArea",
        "MasVnrArea",
        "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
        "BsmtFullBath", "BsmtHalfBath",
    ]
    for col in zero_cols:
        for df in (train, test):
            if col in df.columns:
                df[col] = df[col].fillna(0)

    # LotFrontage — median by Neighborhood (most accurate proxy)
    for df in (train, test):
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median())
        )
    # Fallback for neighborhoods only in test
    for df in (train, test):
        df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    # Mode imputation for remaining low-missing columns
    mode_cols = [
        "MSZoning", "Electrical", "KitchenQual",
        "Exterior1st", "Exterior2nd", "SaleType", "Functional",
    ]
    for col in mode_cols:
        mode_val = train[col].mode()[0] if col in train.columns else None
        for df in (train, test):
            if col in df.columns and mode_val is not None:
                df[col] = df[col].fillna(mode_val)

    print("[impute_missing] Done.")
    return train, test


# ── 4. Rare Category Consolidation ─────────────────────────────────────
def fix_rare_categories(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cat_cols: list[str],
    threshold: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Replace category values that appear in < threshold fraction of training rows
    with the string "Other".

    threshold: default 0.01 = 1% of training set.
    """
    train = train.copy()
    test  = test.copy()
    n = len(train)

    for col in cat_cols:
        if col not in train.columns:
            continue
        freq = train[col].value_counts(normalize=True)
        rare = freq[freq < threshold].index.tolist()
        if rare:
            train[col] = train[col].replace(rare, "Other")
            test[col]  = test[col].replace(rare, "Other") if col in test.columns else test[col]

    print(f"[fix_rare_categories] Processed {len(cat_cols)} categorical columns.")
    return train, test


# ── 5. Fix Data Types ───────────────────────────────────────────────────
def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast pseudo-numeric columns to string so they are One-Hot encoded
    in Phase 3, not treated as continuous variables.

    MSSubClass: encodes building type as numeric codes (20, 30, 40, ...)
    MoSold / YrSold: time metadata, not ordinal predictors
    """
    df = df.copy()
    for col in ["MSSubClass", "MoSold", "YrSold"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    print("[fix_dtypes] MSSubClass, MoSold, YrSold → str")
    return df


# ── 6. Cleaning Report ──────────────────────────────────────────────────
def cleaning_report(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Print a concise summary of the cleaned datasets."""
    sep = "─" * 55
    print(f"\n{sep}")
    print("  DATA CLEANING REPORT")
    print(sep)
    print(f"  Train shape : {train.shape[0]:,} rows × {train.shape[1]} columns")
    print(f"  Test  shape : {test.shape[0]:,} rows × {test.shape[1]} columns")

    train_miss = train.isnull().sum().sum()
    test_miss  = test.isnull().sum().sum()
    print(f"  Missing — train : {train_miss}   test : {test_miss}")

    num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = train.select_dtypes(include="object").columns.tolist()
    print(f"  Numerical columns  : {len(num_cols)}")
    print(f"  Categorical columns: {len(cat_cols)}")

    if "SalePrice" in train.columns:
        print(f"  SalePrice skew  : {train['SalePrice'].skew():.3f}")
    print(sep)
