import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import model_utils as mu

st.set_page_config(page_title="Sepsis Risk Demo", page_icon="🩺", layout="centered")

# ---------------------------------------------------------------------------
# Paths — the two artifacts exported by the training notebook must sit next
# to this file (or set these as env vars when deploying).
# ---------------------------------------------------------------------------
MODEL_DIR = os.environ.get("SEPSIS_MODEL_DIR", os.path.dirname(__file__))
DIST_PATH = os.environ.get("SEPSIS_DIST_PATH", os.path.join(os.path.dirname(__file__), "feature_distributions.json"))


@st.cache_resource
def load_artifacts():
    bundle = mu.load_bundle(MODEL_DIR)
    distributions = mu.load_distributions(DIST_PATH)
    return bundle, distributions


@st.cache_resource
def load_artifacts():
    bundle = mu.load_bundle(MODEL_DIR)
    distributions = mu.load_distributions(DIST_PATH)
    return bundle, distributions


st.title("🩺 Sepsis Risk Estimator")
st.caption(
    "Educational demo — an XGBoost pipeline trained on the PhysioNet 2019 "
    "Sepsis Challenge dataset. **Not a medical device, not for clinical use.**"
)

try:
    bundle, distributions = load_artifacts()
except FileNotFoundError as e:
    st.error(str(e))
    st.info(
    "Place `xgb_model.json` and `sepsis_meta.joblib` "
    "(both produced by the training notebook) in this app's directory, then reload."
        )
    st.stop()


feature_cols = bundle["feature_cols"]
model_name = bundle.get("model_name", "model")

with st.expander("ℹ️ How this works", expanded=False):
    st.markdown(
        f"""
The underlying **{model_name}** pipeline expects **{len(feature_cols)} engineered features**
(summary statistics over each patient's ICU stay: mean / std / min / max / most-recent-value /
missing-rate for 34 vitals & labs, plus demographics).

Filling in all {len(feature_cols)} by hand isn't practical for a quick demo, so:
- The **{len(mu.CURATED_INPUTS)} fields below** are the ones a user would realistically know —
  they set the *current / most recent reading* for each variable.
- **Every other feature** (trends, variability, extremes, missingness patterns, and all labs
  not shown below) is **randomly sampled from the real training data's distribution** for
  each prediction.
- Because most of the feature vector is randomized, a single prediction can vary run to run.
  We draw **{{n_samples}} random backgrounds** and report the **average risk** with a range,
  which is far more stable than any single draw.
        """.replace("{n_samples}", "30")
    )

st.subheader("Patient snapshot")

col1, col2 = st.columns(2)
ui_values = {}

with col1:
    ui_values["age"] = st.slider(mu.CURATED_INPUTS["age"]["label"],
                                  mu.CURATED_INPUTS["age"]["min"], mu.CURATED_INPUTS["age"]["max"],
                                  mu.CURATED_INPUTS["age"]["default"])
    ui_values["gender"] = st.radio(mu.CURATED_INPUTS["gender"]["label"],
                                    list(mu.CURATED_INPUTS["gender"]["options"].keys()), horizontal=True)
    ui_values["icu_los"] = st.slider(mu.CURATED_INPUTS["icu_los"]["label"],
                                      mu.CURATED_INPUTS["icu_los"]["min"], mu.CURATED_INPUTS["icu_los"]["max"],
                                      mu.CURATED_INPUTS["icu_los"]["default"])
    ui_values["hr"] = st.slider(mu.CURATED_INPUTS["hr"]["label"],
                                 mu.CURATED_INPUTS["hr"]["min"], mu.CURATED_INPUTS["hr"]["max"],
                                 mu.CURATED_INPUTS["hr"]["default"])
    ui_values["temp"] = st.slider(mu.CURATED_INPUTS["temp"]["label"],
                                   mu.CURATED_INPUTS["temp"]["min"], mu.CURATED_INPUTS["temp"]["max"],
                                   mu.CURATED_INPUTS["temp"]["default"], step=mu.CURATED_INPUTS["temp"]["step"])
    ui_values["resp"] = st.slider(mu.CURATED_INPUTS["resp"]["label"],
                                   mu.CURATED_INPUTS["resp"]["min"], mu.CURATED_INPUTS["resp"]["max"],
                                   mu.CURATED_INPUTS["resp"]["default"])

with col2:
    ui_values["o2sat"] = st.slider(mu.CURATED_INPUTS["o2sat"]["label"],
                                    mu.CURATED_INPUTS["o2sat"]["min"], mu.CURATED_INPUTS["o2sat"]["max"],
                                    mu.CURATED_INPUTS["o2sat"]["default"])
    ui_values["sbp"] = st.slider(mu.CURATED_INPUTS["sbp"]["label"],
                                  mu.CURATED_INPUTS["sbp"]["min"], mu.CURATED_INPUTS["sbp"]["max"],
                                  mu.CURATED_INPUTS["sbp"]["default"])
    ui_values["map"] = st.slider(mu.CURATED_INPUTS["map"]["label"],
                                  mu.CURATED_INPUTS["map"]["min"], mu.CURATED_INPUTS["map"]["max"],
                                  mu.CURATED_INPUTS["map"]["default"])
    ui_values["wbc"] = st.slider(mu.CURATED_INPUTS["wbc"]["label"],
                                  mu.CURATED_INPUTS["wbc"]["min"], mu.CURATED_INPUTS["wbc"]["max"],
                                  mu.CURATED_INPUTS["wbc"]["default"], step=mu.CURATED_INPUTS["wbc"]["step"])
    ui_values["lactate"] = st.slider(mu.CURATED_INPUTS["lactate"]["label"],
                                      mu.CURATED_INPUTS["lactate"]["min"], mu.CURATED_INPUTS["lactate"]["max"],
                                      mu.CURATED_INPUTS["lactate"]["default"], step=mu.CURATED_INPUTS["lactate"]["step"])

st.divider()
n_samples = st.slider("Number of background draws to average over", 5, 100, 30, step=5,
                       help="More draws = more stable estimate, but slower.")

predict_clicked = st.button("🔬 Estimate Sepsis Risk", type="primary", width="stretch")

if predict_clicked:
    feature_values = mu.user_inputs_to_feature_values(ui_values)
    with st.spinner(f"Running {n_samples} random-background predictions..."):
        probs = mu.predict_risk_distribution(
            bundle, feature_values, feature_cols, distributions,
            n_samples=n_samples, base_seed=None
        )
    mean_prob = float(probs.mean())
    band, color = mu.risk_band(mean_prob)

    st.subheader("Result")
    r1, r2, r3 = st.columns(3)
    r1.metric("Average predicted risk", f"{mean_prob:.1%}")
    r2.metric("Risk band", band)
    r3.metric("Range across draws", f"{probs.min():.1%} – {probs.max():.1%}")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=mean_prob * 100,
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 20], 'color': '#e8f5e9'},
                {'range': [20, 50], 'color': '#fff8e1'},
                {'range': [50, 100], 'color': '#ffebee'},
            ],
        },
        title={'text': "Sepsis Risk"},
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        f"**{band} risk** ({mean_prob:.1%} average across {n_samples} draws). "
        "This reflects your entered values plus randomly sampled background features "
        "consistent with the training population — treat it as illustrative, not diagnostic."
    )

    with st.expander("See distribution across all draws"):
        hist_fig = go.Figure(go.Histogram(x=probs * 100, nbinsx=20, marker_color=color))
        hist_fig.update_layout(
            xaxis_title="Predicted risk (%)", yaxis_title="Count of draws",
            height=280, margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(hist_fig, width="stretch")
        st.caption(
            "Wide spread means the randomly-sampled background features (labs/trends you "
            "didn't specify) are doing a lot of the work — a reminder that this demo fills in "
            "most of the model's inputs randomly rather than from a real patient chart."
        )
else:
    st.info("Set the values above and click **Estimate Sepsis Risk**.")

st.divider()
st.caption(
    "Built on a pipeline trained per the PhysioNet/CinC 2019 Challenge dataset. "
    "For research/educational purposes only — not validated for clinical decision-making."
)
