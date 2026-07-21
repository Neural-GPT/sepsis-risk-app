"""
Core logic for the sepsis risk demo app: loading trained model bundles
(XGBoost, Random Forest, Logistic Regression, KNN), sampling realistic
"background" values for features the user didn't supply, and assembling
a single-row feature vector for prediction.

Kept separate from app.py / pages/*.py so it can be tested independently.
"""
import json
import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

# ---------------------------------------------------------------------------
# Model registry — maps a display name to how it's loaded and used.
# XGBoost is split (native JSON booster + separate preprocessor) because its
# pickle format isn't cross-platform safe (see project history). The other
# three are pure sklearn Pipelines and pickle reliably as a single file.
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "xgboost": {
        "label": "XGBoost",
        "kind": "xgb_native",
        "booster_file": "xgb_model.json",
        "preprocessor_file": "xgb_preprocessor.joblib",
    },
    "random_forest": {
        "label": "Random Forest",
        "kind": "sklearn_pipeline",
        "pipeline_file": "rf_pipeline.joblib",
    },
    "logistic_regression": {
        "label": "Logistic Regression",
        "kind": "sklearn_pipeline",
        "pipeline_file": "logreg_pipeline.joblib",
    },
}
DEFAULT_MODEL_KEY = "xgboost"

# ---------------------------------------------------------------------------
# Curated user-facing inputs (unchanged from the original single-model app).
# Each entry maps a short UI key -> the engineered feature column it feeds
# (the "_last" slot, i.e. "current reading"), plus display metadata and a
# `help` string used for the hover-tooltip explainability feature.
# ---------------------------------------------------------------------------
CURATED_INPUTS = {
    "age": {
        "label": "Age (years)", "feature_col": "Age", "widget": "slider",
        "min": 18, "max": 100, "default": 60, "step": 1,
        "help": "Patient age in years. Sepsis risk and mortality generally increase with age.",
    },
    "gender": {
        "label": "Sex", "feature_col": "Gender", "widget": "radio",
        "options": {"Male": 1, "Female": 0}, "default": "Male",
        "help": "Biological sex, as recorded in the ICU admission data.",
    },
    "icu_los": {
        "label": "Hours in ICU so far", "feature_col": "ICU_LOS", "widget": "slider",
        "min": 1, "max": 200, "default": 24, "step": 1,
        "help": "Length of stay in the ICU so far, in hours. Longer stays correlate with higher cumulative risk exposure.",
    },
    "hr": {
        "label": "Heart Rate (bpm)", "feature_col": "HR_last", "widget": "slider",
        "min": 30, "max": 200, "default": 85, "step": 1,
        "help": "Most recent heart rate reading. Sustained tachycardia (>90 bpm) is a common early sepsis indicator.",
    },
    "temp": {
        "label": "Temperature (\u00b0C)", "feature_col": "Temp_last", "widget": "slider",
        "min": 33.0, "max": 41.0, "default": 37.0, "step": 0.1,
        "help": "Most recent body temperature. Both fever (>38.3\u00b0C) and hypothermia (<36\u00b0C) are Sepsis-3 criteria.",
    },
    "resp": {
        "label": "Respiratory Rate (breaths/min)", "feature_col": "Resp_last", "widget": "slider",
        "min": 5, "max": 45, "default": 18, "step": 1,
        "help": "Most recent respiratory rate. A rate above 22 breaths/min is part of the quick SOFA (qSOFA) criteria.",
    },
    "o2sat": {
        "label": "O2 Saturation (%)", "feature_col": "O2Sat_last", "widget": "slider",
        "min": 60, "max": 100, "default": 97, "step": 1,
        "help": "Most recent peripheral oxygen saturation (SpO2). Low values can indicate respiratory compromise linked to sepsis.",
    },
    "sbp": {
        "label": "Systolic BP (mmHg)", "feature_col": "SBP_last", "widget": "slider",
        "min": 50, "max": 220, "default": 120, "step": 1,
        "help": "Most recent systolic blood pressure. Systolic BP \u2264100 mmHg is part of the qSOFA criteria.",
    },
    "map": {
        "label": "Mean Arterial Pressure (mmHg)", "feature_col": "MAP_last", "widget": "slider",
        "min": 30, "max": 150, "default": 80, "step": 1,
        "help": "Most recent mean arterial pressure. MAP below 65 mmHg often signals inadequate organ perfusion.",
    },
    "wbc": {
        "label": "White Blood Cell Count (x10\u00b3/\u00b5L)", "feature_col": "WBC_last", "widget": "slider",
        "min": 0.0, "max": 40.0, "default": 8.0, "step": 0.1,
        "help": "Most recent white blood cell count. Both very high (>12) and very low (<4) counts are Sepsis-related SIRS criteria.",
    },
    "lactate": {
        "label": "Lactate (mmol/L)", "feature_col": "Lactate_last", "widget": "slider",
        "min": 0.0, "max": 15.0, "default": 1.2, "step": 0.1,
        "help": "Most recent blood lactate level. Elevated lactate (>2 mmol/L) indicates tissue hypoperfusion and is a key sepsis severity marker.",
    },
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def available_models():
    """Return {key: label} for every model present on disk in model_dir."""
    return {k: v["label"] for k, v in MODEL_REGISTRY.items()}


def load_model_bundle(model_dir, model_key):
    """Load a single model's artifacts (booster/pipeline) by registry key."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key '{model_key}'.")
    spec = MODEL_REGISTRY[model_key]

    if spec["kind"] == "xgb_native":
        booster_path = os.path.join(model_dir, spec["booster_file"])
        prep_path = os.path.join(model_dir, spec["preprocessor_file"])
        if not os.path.exists(booster_path) or not os.path.exists(prep_path):
            raise FileNotFoundError(
                f"XGBoost artifacts not found ({spec['booster_file']}, {spec['preprocessor_file']})."
            )
        booster = xgb.Booster()
        booster.load_model(booster_path)
        preprocessor = joblib.load(prep_path)
        return {"kind": "xgb_native", "booster": booster, "preprocessor": preprocessor}

    else:  # sklearn_pipeline
        pipe_path = os.path.join(model_dir, spec["pipeline_file"])
        if not os.path.exists(pipe_path):
            raise FileNotFoundError(f"Model file not found: {spec['pipeline_file']}.")
        pipeline = joblib.load(pipe_path)
        return {"kind": "sklearn_pipeline", "pipeline": pipeline}


def load_bundle(model_dir):
    """
    Load shared metadata (feature columns, evaluation info) plus the
    default model. Individual model switches are handled by
    load_model_bundle() + get_or_load_model() at request time.
    """
    meta_path = os.path.join(model_dir, "sepsis_meta.joblib")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Metadata file not found at '{meta_path}'. "
            "Make sure sepsis_meta.joblib is in the app directory."
        )
    meta = joblib.load(meta_path)
    return {
        "feature_cols": meta["feature_cols"],
        "model_name": meta.get("model_name", "model"),
        "metrics": meta.get("metrics"),
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


def load_evaluation(eval_path):
    """Load the combined model_evaluation.json (metrics + ROC/PR curves for all models)."""
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"Evaluation file not found at '{eval_path}'.")
    with open(eval_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Feature sampling / assembly (model-agnostic)
# ---------------------------------------------------------------------------
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

    user_values: dict of {feature_col_name: value} for curated inputs.
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


# ---------------------------------------------------------------------------
# Prediction (dispatches on model "kind")
# ---------------------------------------------------------------------------
def predict_risk(model_bundle, row_df):
    """Return sepsis probability (float 0-1) for a single-row feature DataFrame,
    regardless of which model kind is loaded."""
    if model_bundle["kind"] == "xgb_native":
        X_transformed = model_bundle["preprocessor"].transform(row_df)
        dmatrix = xgb.DMatrix(X_transformed)
        proba = model_bundle["booster"].predict(dmatrix)
        return float(proba[0])
    else:  # sklearn_pipeline
        proba = model_bundle["pipeline"].predict_proba(row_df)[:, 1]
        return float(proba[0])


def predict_risk_distribution(model_bundle, feature_values, feature_cols, distributions,
                               n_samples=30, base_seed=None):
    """
    Draw `n_samples` independent random backgrounds (same user inputs each time)
    and return the array of predicted probabilities, for whichever model is loaded.
    """
    rng = np.random.default_rng(base_seed)
    seeds = rng.integers(0, 2**31 - 1, size=n_samples)
    probs = []
    for s in seeds:
        row_df, _ = build_feature_row(feature_values, feature_cols, distributions, seed=int(s))
        probs.append(predict_risk(model_bundle, row_df))
    return np.array(probs)


def risk_band(prob):
    """Map a probability to a coarse risk label + color for the UI."""
    if prob < 0.2:
        return "Low", "#2e7d32"
    elif prob < 0.5:
        return "Moderate", "#f9a825"
    else:
        return "High", "#c62828"
