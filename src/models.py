"""
src/models.py
---------------
Advanced ensemble architectures, hyperparameter tuning, blending,
and feature importance extraction for the House Price Predictor project.

Classes & Functions:
    BlendedRegressor            — scikit-learn compatible weighted ensemble estimator
    get_advanced_models         — instantiate RF, GBM, XGBoost, and LightGBM
    tune_xgboost                — randomized search over boosting hyperparameters
    build_stacking_regressor    — multi-paradigm StackingRegressor with meta-learner
    get_feature_importances     — extract and rank feature importances from tree models
"""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


# ── 1. Custom Blended Regressor ─────────────────────────────────────────
class BlendedRegressor(BaseEstimator, RegressorMixin):
    """
    Weighted linear blend of diverse regression models.
    Combines linear regularization with tree gradient boosting for robust generalization.
    """

    def __init__(self, models: dict[str, Any] | None = None, weights: dict[str, float] | None = None):
        self.models = models or {}
        self.weights = weights or {}

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray):
        for name, model in self.models.items():
            model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        total_weight = sum(self.weights.values())
        weighted_preds = np.zeros(len(X))
        for name, model in self.models.items():
            w = self.weights.get(name, 1.0) / total_weight
            weighted_preds += w * model.predict(X)
        return weighted_preds


# ── 2. Advanced Model Instantiation ─────────────────────────────────────
def get_advanced_models(random_state: int = 42) -> dict[str, Any]:
    """
    Instantiate tuned ensemble architectures for benchmarking.
    """
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            min_samples_leaf=3,
            random_state=random_state,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=31,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=random_state,
            verbosity=-1,
            n_jobs=-1,
        ),
    }


# ── 3. Hyperparameter Tuning ────────────────────────────────────────────
def tune_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 15,
    cv: int = 5,
    random_state: int = 42,
) -> tuple[XGBRegressor, dict[str, Any]]:
    """
    Tune XGBoost hyperparameters via RandomizedSearchCV.
    """
    param_dist = {
        "n_estimators": [300, 500, 700],
        "learning_rate": [0.01, 0.03, 0.05],
        "max_depth": [3, 4, 5],
        "subsample": [0.6, 0.7, 0.8],
        "colsample_bytree": [0.6, 0.7, 0.8],
        "reg_alpha": [0.01, 0.1, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0],
    }

    base_xgb = XGBRegressor(random_state=random_state, n_jobs=-1)
    search = RandomizedSearchCV(
        base_xgb,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X, y)
    print(f"[tune_xgboost] Best CV RMSE: {-search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_


# ── 4. Stacking Regressor ───────────────────────────────────────────────
def build_stacking_regressor(random_state: int = 42) -> StackingRegressor:
    """
    Construct a heterogeneous 4-model StackingRegressor:
    Base learners: Lasso, Ridge, XGBoost, LightGBM.
    Meta-learner: Ridge(alpha=10.0) combining out-of-fold predictions.
    """
    estimators = [
        ("lasso", make_pipeline(RobustScaler(), Lasso(alpha=0.00049, random_state=random_state, max_iter=10000))),
        ("ridge", make_pipeline(RobustScaler(), Ridge(alpha=16.0, random_state=random_state))),
        ("xgb", XGBRegressor(n_estimators=450, learning_rate=0.03, max_depth=4, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, random_state=random_state, n_jobs=-1)),
        ("lgb", LGBMRegressor(n_estimators=450, learning_rate=0.03, max_depth=5, num_leaves=31, subsample=0.7, colsample_bytree=0.7, random_state=random_state, verbosity=-1, n_jobs=-1)),
    ]

    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=10.0, random_state=random_state),
        cv=5,
        n_jobs=-1,
    )
    return stack


# ── 5. Feature Importance Extraction ────────────────────────────────────
def get_feature_importances(
    model: Any,
    feature_names: list[str] | pd.Index,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Extract, normalize, and sort feature importances from a tree-based estimator.
    """
    if hasattr(model, "feature_importances_"):
        raw_imp = model.feature_importances_
    elif hasattr(model, "named_steps"):
        step = model.steps[-1][1]
        raw_imp = getattr(step, "feature_importances_", None)
    else:
        raw_imp = None

    if raw_imp is None:
        raise ValueError(f"Model {type(model)} does not expose feature_importances_")

    norm_imp = (raw_imp / raw_imp.sum()) * 100
    df_imp = pd.DataFrame({
        "Feature": list(feature_names),
        "Importance_Pct": norm_imp,
    }).sort_values(by="Importance_Pct", ascending=False).reset_index(drop=True)

    return df_imp.head(top_n)
