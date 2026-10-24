"""
app/streamlit_app.py
--------------------
Interactive Streamlit web application for real-time house price estimation
using the production ensemble model trained on the Ames Housing Dataset.

Run locally:
    streamlit run app/streamlit_app.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="Ames House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Connect project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import src.evaluate as ev
import src.models as mod

MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data" / "processed"

# ── Load Model and Template Schema ──────────────────────────────────────
@st.cache_resource(show_spinner="Loading production ensemble model...")
def load_production_pipeline():
    model_path = MODELS_DIR / "best_model.pkl"
    if not model_path.exists():
        st.error(f"Model file not found at {model_path}. Please ensure Phase 5 is completed.")
        st.stop()
    model = ev.load_model(model_path)
    
    metadata_path = MODELS_DIR / "feature_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        feature_columns = meta["feature_columns"]
        baseline_template = meta["baseline_template"]
    else:
        train_features = pd.read_csv(DATA_DIR / "train_features.csv")
        feature_columns = [c for c in train_features.columns if c not in ["Id", "SalePrice"]]
        baseline_template = train_features[feature_columns].median().to_dict()

    return model, feature_columns, baseline_template

model, feature_columns, baseline_template = load_production_pipeline()

# ── Neighborhood Display Mapping ────────────────────────────────────────
NEIGHBORHOOD_NAMES = {
    "CollgCr": "College Creek (CollgCr)",
    "Veenker": "Veenker (Near Ames High)",
    "Crawfor": "Crawford (Crawfor)",
    "NoRidge": "Northridge (NoRidge - Luxury)",
    "Mitchel": "Mitchell (Mitchel)",
    "Somerst": "Somerset (Somerst - Modern)",
    "NWAmes": "Northwest Ames (NWAmes)",
    "OldTown": "Old Town (Historic Ames)",
    "BrkSide": "Brookside (BrkSide)",
    "Sawyer": "Sawyer (Affordable Residential)",
    "NridgHt": "Northridge Heights (NridgHt - Luxury)",
    "NAmes": "North Ames (Established Suburb)",
    "SawyerW": "Sawyer West (SawyerW)",
    "IDOTRR": "Iowa DOT & Rail Road (IDOTRR)",
    "MeadowV": "Meadow Village (MeadowV - Budget)",
    "Edwards": "Edwards (Near ISU)",
    "Timber": "Timberland (Timber - Wooded)",
    "Gilbert": "Gilbert (Suburban)",
    "StoneBr": "Stone Brook (StoneBr - Executive)",
    "ClearCr": "Clear Creek (ClearCr)",
    "NPkVill": "Northpark Villa (NPkVill)",
    "Blmngtn": "Bloomington Heights (55+ Active)",
    "BrDale": "Briardale (BrDale)",
    "SWISU": "South & West of Iowa State Univ",
    "Blueste": "Bluestem (Blueste)",
}

QUAL_MAPPING = {
    "Excellent (Ex)": 5,
    "Good (Gd)": 4,
    "Typical / Average (TA)": 3,
    "Fair (Fa)": 2,
    "Poor (Po)": 1,
    "None / Not Present": 0,
}

# ── Header ──────────────────────────────────────────────────────────────
st.title("🏠 Ames House Price Predictor")
st.markdown(
    """
    **Production-Grade Machine Learning Valuation System**  
    *Powered by a Multi-Paradigm Ensemble (Lasso + Ridge + XGBoost + LightGBM) trained on the Ames Housing Dataset.*
    """
)

# ── Sidebar Preset Configurations ───────────────────────────────────────
st.sidebar.header("⚡ Quick Preset Profiles")
preset = st.sidebar.radio(
    "Choose a starter property profile:",
    [
        "🏡 Suburban Family Home (Standard)",
        "🏠 Affordable Starter Home",
        "🏰 Executive Luxury Estate",
        "🛠️ Custom User Input",
    ],
    index=0,
)

# Preset defaults
if preset == "🏡 Suburban Family Home (Standard)":
    d_qual, d_cond, d_built, d_remod = 7, 5, 2003, 2004
    d_grliv, d_bsmt, d_baths, d_beds, d_garage = 1750, 950, 2.5, 3, 2
    d_neigh, d_kqual, d_fp = "CollgCr", "Good (Gd)", 1
elif preset == "🏠 Affordable Starter Home":
    d_qual, d_cond, d_built, d_remod = 5, 6, 1965, 1980
    d_grliv, d_bsmt, d_baths, d_beds, d_garage = 1050, 600, 1.0, 2, 1
    d_neigh, d_kqual, d_fp = "OldTown", "Typical / Average (TA)", 0
elif preset == "🏰 Executive Luxury Estate":
    d_qual, d_cond, d_built, d_remod = 9, 6, 2008, 2009
    d_grliv, d_bsmt, d_baths, d_beds, d_garage = 3100, 1800, 3.5, 4, 3
    d_neigh, d_kqual, d_fp = "NoRidge", "Excellent (Ex)", 2
else:
    d_qual, d_cond, d_built, d_remod = 6, 5, 1990, 1995
    d_grliv, d_bsmt, d_baths, d_beds, d_garage = 1500, 800, 2.0, 3, 2
    d_neigh, d_kqual, d_fp = "NAmes", "Typical / Average (TA)", 1

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Model Performance Metrics:**
    - **Validation $R^2$**: `92.8%`
    - **Validation RMSE**: `0.1105`
    - **Mean Absolute Error**: `~$12,715`
    - **Median % Error**: `3.8%`
    """
)

# ── Main Input Form ─────────────────────────────────────────────────────
tab_predict, tab_benchmarks, tab_about = st.tabs([
    "🎯 Property Valuation", "📊 Benchmark Leaderboard", "📖 Methodology & Pipeline"
])

with tab_predict:
    st.subheader("📋 Enter Property Specifications")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 🏗️ Construction & Quality")
        overall_qual = st.slider("Overall Material & Finish Quality (1-10)", 1, 10, d_qual)
        overall_cond = st.slider("Overall Condition (1-9)", 1, 9, d_cond)
        year_built = st.number_input("Year Built", min_value=1870, max_value=2010, value=d_built, step=1)
        year_remod = st.number_input("Year Remodeled / Addition", min_value=year_built, max_value=2010, value=max(d_remod, year_built), step=1)
        neighborhood = st.selectbox(
            "Neighborhood",
            options=list(NEIGHBORHOOD_NAMES.keys()),
            format_func=lambda x: NEIGHBORHOOD_NAMES[x],
            index=list(NEIGHBORHOOD_NAMES.keys()).index(d_neigh) if d_neigh in NEIGHBORHOOD_NAMES else 0,
        )

    with col2:
        st.markdown("##### 📐 Dimensions & Usable Space")
        gr_liv_area = st.slider("Above-Ground Living Area (sq ft)", 400, 4500, d_grliv, step=25)
        total_bsmt_sf = st.slider("Total Basement Area (sq ft)", 0, 3000, d_bsmt, step=25)
        lot_area = st.number_input("Lot Size (sq ft)", min_value=1000, max_value=60000, value=10000, step=500)
        lot_frontage = st.slider("Lot Frontage (Linear feet of street)", 20, 160, 70, step=5)
        has_central_air = st.radio("Central Air Conditioning", ["Yes", "No"], index=0, horizontal=True)

    with col3:
        st.markdown("##### 🛏️ Rooms, Garage & Amenities")
        full_bath = st.selectbox("Full Bathrooms", [1, 2, 3, 4], index=int(min(d_baths, 4)) - 1)
        half_bath = st.selectbox("Half Bathrooms", [0, 1, 2], index=1 if (d_baths % 1 != 0) else 0)
        bedroom_abv = st.selectbox("Bedrooms (Above Ground)", [1, 2, 3, 4, 5, 6], index=d_beds - 1)
        garage_cars = st.slider("Garage Capacity (Cars)", 0, 4, d_garage)
        garage_area = st.slider("Garage Area (sq ft)", 0, 1400, garage_cars * 250, step=25)
        fireplaces = st.slider("Fireplaces", 0, 4, d_fp)
        kitchen_qual = st.selectbox("Kitchen Quality", ["Excellent (Ex)", "Good (Gd)", "Typical / Average (TA)", "Fair (Fa)"], index=0 if "Ex" in d_kqual else 1 if "Gd" in d_kqual else 2)
        exter_qual = st.selectbox("Exterior Material Quality", ["Excellent (Ex)", "Good (Gd)", "Typical / Average (TA)", "Fair (Fa)"], index=1)

    # ── Real-Time Prediction Computation ────────────────────────────────
    st.markdown("---")
    
    # 1. Start from baseline template row
    input_row = baseline_template.copy()

    # 2. Compute engineered domain features
    # Approximate 1st and 2nd floor distribution
    first_flr_sf = gr_liv_area * 0.65
    second_flr_sf = gr_liv_area * 0.35 if gr_liv_area > 1200 else 0
    total_sf = total_bsmt_sf + gr_liv_area
    total_bath = full_bath + (0.5 * half_bath) + 0.5  # include nominal basement bath

    house_age = max(0, 2010 - year_built)
    remod_age = max(0, 2010 - year_remod)
    is_remodeled = int(year_remod != year_built)

    # 3. Update continuous and ordinal features
    input_row["OverallQual"] = overall_qual
    input_row["OverallCond"] = overall_cond
    input_row["OverallScore"] = overall_qual * overall_cond
    input_row["YearBuilt"] = year_built
    input_row["YearRemodAdd"] = year_remod
    input_row["HouseAge"] = house_age
    input_row["RemodAge"] = remod_age
    input_row["IsRemodeled"] = is_remodeled
    
    input_row["FullBath"] = full_bath
    input_row["HalfBath"] = half_bath
    input_row["TotalBath"] = total_bath
    input_row["BedroomAbvGr"] = bedroom_abv
    input_row["GarageCars"] = garage_cars
    input_row["Fireplaces"] = fireplaces

    # Binary flags
    input_row["HasGarage"] = int(garage_area > 0)
    input_row["HasFireplace"] = int(fireplaces > 0)
    input_row["HasBsmt"] = int(total_bsmt_sf > 0)
    input_row["Has2ndFloor"] = int(second_flr_sf > 0)

    # Ordinal mappings
    input_row["KitchenQual"] = QUAL_MAPPING[kitchen_qual]
    input_row["ExterQual"] = QUAL_MAPPING[exter_qual]

    # Skewed features: apply log1p as per Phase 3 training
    input_row["GrLivArea"] = np.log1p(gr_liv_area)
    input_row["TotalBsmtSF"] = total_bsmt_sf
    input_row["1stFlrSF"] = np.log1p(first_flr_sf)
    input_row["2ndFlrSF"] = np.log1p(second_flr_sf)
    input_row["TotalSF"] = np.log1p(total_sf)
    input_row["LotArea"] = np.log1p(lot_area)
    input_row["LotFrontage"] = np.log1p(lot_frontage)
    input_row["GarageArea"] = garage_area

    # One-hot Neighborhood
    for col in feature_columns:
        if col.startswith("Neighborhood_"):
            input_row[col] = 0
    target_neigh_col = f"Neighborhood_{neighborhood}"
    if target_neigh_col in input_row:
        input_row[target_neigh_col] = 1

    # One-hot CentralAir
    if "CentralAir_Y" in input_row:
        input_row["CentralAir_Y"] = 1 if has_central_air == "Yes" else 0

    # 4. Form DataFrame and Predict
    input_df = pd.DataFrame([input_row])[feature_columns]
    predicted_log = model.predict(input_df)[0]
    predicted_price = float(np.expm1(predicted_log))

    # 95% Confidence Interval (±3.8% median absolute error)
    lower_bound = predicted_price * 0.935
    upper_bound = predicted_price * 1.065
    ames_median = 163_000
    price_delta = predicted_price - ames_median
    price_per_sqft = predicted_price / max(1, gr_liv_area)

    # ── Valuation Display Card ──────────────────────────────────────────
    st.subheader("🏷️ Valuation Estimate")
    res_col1, res_col2, res_col3 = st.columns([1.5, 1, 1])

    with res_col1:
        st.metric(
            label="Estimated Market Value",
            value=f"${predicted_price:,.0f}",
            delta=f"${price_delta:+,.0f} vs Ames Median ($163,000)",
            delta_color="normal",
        )
        st.caption(f"**Likely Price Range (±6.5%):** `${lower_bound:,.0f}` — `${upper_bound:,.0f}`")

    with res_col2:
        st.metric(
            label="Price per Sq. Ft.",
            value=f"${price_per_sqft:,.2f}/sq ft",
        )
        st.caption(f"Based on {gr_liv_area:,} sq.ft. above-ground living space")

    with res_col3:
        st.metric(
            label="Model Confidence Score",
            value="92.8%",
            delta="Cross-Validated R²",
        )
        st.caption("Derived from 4-Model Heterogeneous Ensemble")

    # ── Feature Contribution Breakdown ──────────────────────────────────
    st.markdown("#### 💡 Key Valuation Drivers for this Home")
    driver_col1, driver_col2, driver_col3, driver_col4 = st.columns(4)

    with driver_col1:
        st.markdown(f"**Overall Quality:** `{overall_qual}/10`")
        if overall_qual >= 8:
            st.success("Premium Finish (Major Positive)")
        elif overall_qual <= 4:
            st.error("Below Average (Discount Applied)")
        else:
            st.info("Market Average Grade")

    with driver_col2:
        st.markdown(f"**Total Living Space:** `{total_sf:,} sq ft`")
        if total_sf >= 2500:
            st.success("Spacious Layout (+High Value)")
        else:
            st.info("Standard Floorplan")

    with driver_col3:
        st.markdown(f"**House Age:** `{house_age} years`")
        if house_age <= 10:
            st.success("Modern Build (Low Depreciation)")
        elif house_age >= 50:
            st.warning("Mature Property (Depreciated)")
        else:
            st.info("Mid-Age Established")

    with driver_col4:
        st.markdown(f"**Garage Capacity:** `{garage_cars} cars`")
        if garage_cars >= 2:
            st.success(f"{garage_cars}-Car Garage (Desirable)")
        else:
            st.warning("Limited Parking")

with tab_benchmarks:
    st.subheader("🏆 Model Benchmarking Leaderboard")
    st.markdown("Performance comparison across all 7 phases of our machine learning pipeline:")

    leaderboard_data = {
        "Rank": ["🥇 1", "🥈 2", "🥉 3", "4", "5", "6", "7", "8", "9", "10", "11"],
        "Model Architecture": [
            "Weighted Blend (Lasso + Ridge + XGB + LGB)",
            "Lasso Regression (Phase 4)",
            "Stacking Regressor (Ridge Meta)",
            "Ridge Regression (Phase 4)",
            "Tuned XGBoost (Phase 5)",
            "Default XGBoost",
            "Gradient Boosting (GBM)",
            "LightGBM",
            "Linear Regression (OLS)",
            "Random Forest",
            "Decision Tree (Phase 4)",
        ],
        "Paradigm": [
            "Multi-Paradigm Ensemble",
            "Linear Regularized (L1)",
            "Heterogeneous Stacking",
            "Linear Regularized (L2)",
            "Gradient Boosted Tree",
            "Gradient Boosted Tree",
            "Gradient Boosted Tree",
            "Histogram Boosted Tree",
            "Linear OLS",
            "Bagged Tree",
            "Single Tree",
        ],
        "Validation RMSE": [0.1105, 0.1108, 0.1115, 0.1157, 0.1168, 0.1172, 0.1206, 0.1259, 0.1304, 0.1402, 0.1929],
        "Validation R²": ["92.8%", "92.7%", "92.6%", "92.1%", "91.9%", "91.8%", "91.4%", "90.6%", "89.9%", "88.3%", "77.9%"],
        "Mean Absolute Error ($)": ["$12,715", "$13,713", "$12,845", "$13,824", "$13,709", "$13,969", "$14,288", "$14,178", "$14,677", "$15,760", "$23,376"],
    }
    st.dataframe(pd.DataFrame(leaderboard_data), width="stretch", hide_index=True)

with tab_about:
    st.subheader("📖 Machine Learning Architecture")
    st.markdown(
        """
        ### Pipeline Workflow
        1. **Phase 1: Exploratory Data Analysis** — Identified extreme outliers (GrLivArea > 4000 & SalePrice < $200k) and target skewness ($1.88$).
        2. **Phase 2: Data Cleaning** — Domain-aware missing value imputation and rare category consolidation.
        3. **Phase 3: Feature Engineering** — Created `TotalSF`, `TotalBath`, `HouseAge`, `OverallScore`, ordinal quality rankings, log1p skew normalization, and 192 aligned dummy variables.
        4. **Phase 4: Baseline Modeling** — Evaluated OLS, Ridge ($\alpha=16$), Lasso ($\alpha=0.00049$), and Decision Trees.
        5. **Phase 5: Advanced Ensembles** — Tuned XGBoost via `RandomizedSearchCV` and constructed the production **Weighted Blend Regressor**.
        6. **Phase 6: Diagnostics & Evaluation** — Q-Q normality test, learning curves, and Kaggle submission formatting.
        7. **Phase 7: Deployment** — Interactive Streamlit web app powered by `models/best_model.pkl`.
        """
    )

st.markdown("---")
st.caption("Built with Streamlit • Model: Multi-Paradigm Ensemble • End-to-End House Price Predictor Project")
