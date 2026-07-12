import json
import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

# ---------------------------------------------------------------------------
# Curated user-facing inputs.
#
# Each entry maps a short UI key -> the engineered feature column it feeds
# (the "_last" slot, i.e. "current reading"), plus display metadata for the
# Streamlit widget. Everything else in the pipeline's feature vector is
# randomly sampled from the training distribution at predict time.
# ---------------------------------------------------------------------------
CURATED_INPUTS = {
    "age": {
        "label": "Age (years)",
        "feature_col": "Age",
        "widget": "slider",
        "min": 18, "max": 100, "default": 60, "step": 1,
    },
    "gender": {
        "label": "Sex",
        "feature_col": "Gender",
        "widget": "radio",
        "options": {"Male": 1, "Female": 0},
        "default": "Male",
    },
    "icu_los": {
        "label": "Hours in ICU so far",
        "feature_col": "ICU_LOS",
        "widget": "slider",
        "min": 1, "max": 200, "default": 24, "step": 1,
    },
    "hr": {
        "label": "Heart Rate (bpm)",
        "feature_col": "HR_last",
        "widget": "slider",
        "min": 30, "max": 200, "default": 85, "step": 1,
    },
    "temp": {
        "label": "Temperature (°C)",
        "feature_col": "Temp_last",
        "widget": "slider",
        "min": 33.0, "max": 41.0, "default": 37.0, "step": 0.1,
    },
    "resp": {
        "label": "Respiratory Rate (breaths/min)",
        "feature_col": "Resp_last",
        "widget": "slider",
        "min": 5, "max": 45, "default": 18, "step": 1,
    },
    "o2sat": {
        "label": "O2 Saturation (%)",
        "feature_col": "O2Sat_last",
        "widget": "slider",
        "min": 60, "max": 100, "default": 97, "step": 1,
    },
    "sbp": {
        "label": "Systolic BP (mmHg)",
        "feature_col": "SBP_last",
        "widget": "slider",
        "min": 50, "max": 220, "default": 120, "step": 1,
    },
    "map": {
        "label": "Mean Arterial Pressure (mmHg)",
        "feature_col": "MAP_last",
        "widget": "slider",
        "min": 30, "max": 150, "default": 80, "step": 1,
    },
    "wbc": {
        "label": "White Blood Cell Count (x10\u00b3/\u00b5L)",
        "feature_col": "WBC_last",
        "widget": "slider",
        "min": 0.0, "max": 40.0, "default": 8.0, "step": 0.1,
    },
    "lactate": {
        "label": "Lactate (mmol/L)",
        "feature_col": "Lactate_last",
        "widget": "slider",
        "min": 0.0, "max": 15.0, "default": 1.2, "step": 0.1,
    },
}


def load_bundle(model_dir):
    """Load the split model artifacts: xgboost native model + joblib metadata."""
    meta_path = os.path.join(model_dir, "sepsis_meta.joblib")
    xgb_path = os.path.join(model_dir, "xgb_model.json")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Metadata file not found at '{meta_path}'. "
            "Make sure sepsis_meta.joblib is in the app directory."
        )
    if not os.path.exists(xgb_path):
        raise FileNotFoundError(
            f"Model file not found at '{xgb_path}'. "
            "Make sure xgb_model.json is in the app directory."
        )

    meta = joblib.load(meta_path)
    booster = xgb.Booster()
    booster.load_model(xgb_path)

    return {
        "preprocessor": meta["preprocessor"],
        "booster": booster,
        "feature_cols": meta["feature_cols"],
        "model_name": meta["model_name"],
        "metrics": meta["metrics"],
    }

def load_distributions(dist_path):
    """Load the per-feature quantile grids exported by the training notebook."""
    if not os.path.exists(dist_path):
        raise FileNotFoundError(
            f"Distributions file not found at '{dist_path}'. "
            "Make sure feature_distributions.json is in the app directory."
        )
    with open(dist_path) as f:
        return json.load(f)


def sample_feature_value(dist_entry, rng):
    """Draw one realistic value for a feature via inverse-CDF interpolation
    over its stored quantile grid."""
    u = rng.uniform(0, 100)
    levels = dist_entry["quantile_levels"]
    values = dist_entry["quantile_values"]
    return float(np.interp(u, levels, values))


def build_feature_row(user_values, feature_cols, distributions, seed=None):
    """
    Assemble a single-row feature vector matching `feature_cols` order.

    user_values: dict of {feature_col_name: value} for curated inputs
                 (already translated from UI keys, e.g. 'HR_last': 85).
    Everything else in feature_cols is sampled from `distributions`.
    Returns (pd.DataFrame with one row, dict of {col: was_randomly_sampled}).
    """
    rng = np.random.default_rng(seed)
    row = {}
    was_sampled = {}

    for col in feature_cols:
        if col in user_values:
            row[col] = user_values[col]
            was_sampled[col] = False
        elif col in distributions:
            row[col] = sample_feature_value(distributions[col], rng)
            was_sampled[col] = True
        else:
            # No distribution info available (shouldn't normally happen) -> leave NaN,
            # the pipeline's imputer will fill it.
            row[col] = np.nan
            was_sampled[col] = True

    df = pd.DataFrame([row], columns=feature_cols)
    return df, was_sampled


def user_inputs_to_feature_values(ui_values):
    """Translate {ui_key: raw_widget_value} -> {feature_col: numeric_value}
    using CURATED_INPUTS metadata (handles the Gender radio mapping)."""
    out = {}
    for key, val in ui_values.items():
        spec = CURATED_INPUTS[key]
        col = spec["feature_col"]
        if spec["widget"] == "radio":
            out[col] = spec["options"][val]
        else:
            out[col] = float(val)
    return out


def predict_risk(bundle, row_df):
    """Return sepsis probability (float 0-1) for a single-row feature DataFrame."""
    X_transformed = bundle["preprocessor"].transform(row_df)
    dmatrix = xgb.DMatrix(X_transformed)
    proba = bundle["booster"].predict(dmatrix)
    return float(proba[0])

def predict_risk_distribution(bundle, feature_values, feature_cols, distributions,
                               n_samples=30, base_seed=None):
    rng = np.random.default_rng(base_seed)
    seeds = rng.integers(0, 2**31 - 1, size=n_samples)
    probs = []
    for s in seeds:
        row_df, _ = build_feature_row(feature_values, feature_cols, distributions, seed=int(s))
        probs.append(predict_risk(bundle, row_df))
    return np.array(probs)

def risk_band(prob):
    """Map a probability to a coarse risk label + color for the UI."""
    if prob < 0.2:
        return "Low", "#2e7d32"
    elif prob < 0.5:
        return "Moderate", "#f9a825"
    else:
        return "High", "#c62828"
