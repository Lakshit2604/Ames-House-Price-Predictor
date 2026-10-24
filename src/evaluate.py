"""
src/evaluate.py
---------------
Reusable model evaluation, benchmarking, and diagnostic functions
for the House Price Predictor project.

Functions:
    compute_metrics         — calculate RMSE (log & $), MAE ($), and R2
    cross_validate_model    — compute K-Fold cross-validation RMSE mean and std
    benchmark_models        — compare a dictionary of models across CV and test splits
    plot_predictions        — plot predicted vs actual prices with ideal line
    plot_residuals          — plot residual scatter and distribution for error analysis
    save_model              — serialize trained model pipeline using joblib
    load_model              — deserialize model pipeline from disk
"""

from pathlib import Path
from typing import Any
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score


# ── 1. Metric Computation ───────────────────────────────────────────────
def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    is_log_scale: bool = True,
) -> dict[str, float]:
    """
    Compute standard regression metrics in both log-price space and real-world dollars.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    if is_log_scale:
        rmse_log = root_mean_squared_error(y_true_arr, y_pred_arr)
        r2 = r2_score(y_true_arr, y_pred_arr)

        # Invert log1p back to dollar values for interpretable business metrics
        y_true_dollars = np.expm1(y_true_arr)
        y_pred_dollars = np.expm1(y_pred_arr)
        rmse_dollars = root_mean_squared_error(y_true_dollars, y_pred_dollars)
        mae_dollars = mean_absolute_error(y_true_dollars, y_pred_dollars)

        return {
            "RMSE_log": rmse_log,
            "R2": r2,
            "RMSE_dollars": rmse_dollars,
            "MAE_dollars": mae_dollars,
        }
    else:
        rmse = root_mean_squared_error(y_true_arr, y_pred_arr)
        mae = mean_absolute_error(y_true_arr, y_pred_arr)
        r2 = r2_score(y_true_arr, y_pred_arr)
        return {"RMSE": rmse, "MAE": mae, "R2": r2}


# ── 2. Cross Validation ─────────────────────────────────────────────────
def cross_validate_model(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    cv: int = 5,
    random_state: int = 42,
) -> tuple[float, float]:
    """
    Perform K-Fold cross validation and return mean and std of RMSE on log-price.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = -cross_val_score(
        model, X, y, cv=kf, scoring="neg_root_mean_squared_error"
    )
    return float(scores.mean()), float(scores.std())


# ── 3. Benchmark Multiple Models ─────────────────────────────────────────
def benchmark_models(
    models: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cv: int = 5,
) -> pd.DataFrame:
    """
    Train and benchmark a dictionary of candidate models.
    Returns a sorted DataFrame ranking models by Validation RMSE.
    """
    records = []
    for name, model in models.items():
        # 1. Cross-validation across training set
        cv_mean, cv_std = cross_validate_model(model, X_train, y_train, cv=cv)

        # 2. Fit on training split and evaluate on holdout validation split
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        metrics = compute_metrics(y_val, val_preds, is_log_scale=True)

        records.append({
            "Model": name,
            "CV_RMSE_Mean": cv_mean,
            "CV_RMSE_Std": cv_std,
            "Val_RMSE_Log": metrics["RMSE_log"],
            "Val_R2": metrics["R2"],
            "Val_RMSE_USD": metrics["RMSE_dollars"],
            "Val_MAE_USD": metrics["MAE_dollars"],
        })

    df_results = pd.DataFrame(records).sort_values(
        by="Val_RMSE_Log", ascending=True
    ).reset_index(drop=True)
    return df_results


# ── 4. Diagnostics & Visualizations ─────────────────────────────────────
def plot_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    model_name: str = "Model",
    save_path: str | Path | None = None,
) -> None:
    """Scatter plot of actual vs predicted house prices with 45-degree parity line."""
    plt.figure(figsize=(8, 6))
    y_true_d = np.expm1(y_true)
    y_pred_d = np.expm1(y_pred)

    sns.scatterplot(x=y_true_d, y=y_pred_d, alpha=0.6, color="#2980b9")
    max_val = max(y_true_d.max(), y_pred_d.max()) * 1.05
    plt.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Perfect Fit (y = x)")

    plt.title(f"{model_name}: Actual vs Predicted Sale Price ($)", fontsize=12, fontweight="bold")
    plt.xlabel("Actual Sale Price ($)", fontsize=11)
    plt.ylabel("Predicted Sale Price ($)", fontsize=11)
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.legend()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_residuals(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    model_name: str = "Model",
    save_path: str | Path | None = None,
) -> None:
    """Residual scatter and distribution plots to check homoscedasticity and normality."""
    residuals = np.array(y_true) - np.array(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.6, color="#8e44ad")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=2)
    axes[0].set_title(f"{model_name}: Residuals vs Predicted (log)", fontweight="bold")
    axes[0].set_xlabel("Predicted log(SalePrice)")
    axes[0].set_ylabel("Residual (Actual - Predicted)")

    # 2. Residual Distribution
    sns.histplot(residuals, kde=True, ax=axes[1], color="#16a085")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[1].set_title(f"{model_name}: Residuals Distribution", fontweight="bold")
    axes[1].set_xlabel("Residual (log-space error)")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()


# ── 5. Model Serialization ──────────────────────────────────────────────
def save_model(model: Any, filepath: str | Path) -> None:
    """Serialize model object using joblib."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"[save_model] Model successfully saved to: {path}")


def load_model(filepath: str | Path) -> Any:
    """Deserialize model object from disk."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    print(f"[load_model] Loaded model from: {path}")
    return model
