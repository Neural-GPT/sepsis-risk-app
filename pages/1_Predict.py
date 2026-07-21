import os
import streamlit as st
import plotly.graph_objects as go

from theme import inject_theme, glass_card, section_header, render_sidebar
import auth_utils as auth
import model_utils as mu
import llm_utils as llm
import db_utils as db

st.set_page_config(page_title="Predict — Sepsis Risk Estimator", page_icon=":material/biotech:", layout="wide")
inject_theme()
auth.require_login()

MODEL_DIR = os.environ.get("SEPSIS_MODEL_DIR", os.path.join(os.path.dirname(__file__), ".."))
DIST_PATH = os.environ.get("SEPSIS_DIST_PATH", os.path.join(MODEL_DIR, "feature_distributions.json"))

with st.sidebar:
    render_sidebar(auth.current_user())
    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    n_samples = st.slider(
        "Background draws", 5, 100, 30, step=5,
        help="Number of randomly-sampled background feature sets averaged per prediction. "
             "More draws = more stable estimate, but slower.",
    )
    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    if st.button("Log Out", width="stretch"):
        auth.logout()
        st.switch_page("app.py")

# -----------------------------------------------------------------------
# Load shared metadata + distributions (cached)
# -----------------------------------------------------------------------
@st.cache_resource
def load_meta_and_dist():
    meta = mu.load_bundle(MODEL_DIR)
    distributions = mu.load_distributions(DIST_PATH)
    return meta, distributions


@st.cache_resource
def load_model(key):
    return mu.load_model_bundle(MODEL_DIR, key)


section_header("biotech", "Sepsis Risk Estimator",
                "Educational demo — not a medical device, not for clinical use.")

try:
    meta, distributions = load_meta_and_dist()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

feature_cols = meta["feature_cols"]

# -----------------------------------------------------------------------
# Model selector — prominent, on the main page
# -----------------------------------------------------------------------
available = mu.available_models()
label_to_key = {v: k for k, v in available.items()}
default_label = available[mu.DEFAULT_MODEL_KEY]

with glass_card("glass-model-select"):
    st.markdown("**Choose a model**")
    selected_label = st.segmented_control(
        "Model", options=list(label_to_key.keys()), default=default_label,
        label_visibility="collapsed",
        help="Switch between trained classifiers. XGBoost generally gives the best "
             "balance of precision and recall for this task.",
    )
    if not selected_label:
        selected_label = default_label
    model_key = label_to_key[selected_label]

try:
    model_bundle = load_model(model_key)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

with st.expander("How this works", icon=":material/info:", expanded=False):
    st.markdown(
        f"""
The selected model (**{available[model_key]}**) expects **{len(feature_cols)}
engineered features** (summary statistics over each patient's ICU stay across 34 vitals &
labs, plus demographics).

- The **{len(mu.CURATED_INPUTS)} fields below** are the ones a user would realistically know.
  Hover the info icon next to each for a clinical explanation of what it measures and why it matters.
- **Every other feature** is randomly sampled from the real training data's distribution for
  each prediction, and results are averaged across multiple draws for a stable estimate.
        """
    )

# -----------------------------------------------------------------------
# Curated inputs, with hover tooltips (help=)
# -----------------------------------------------------------------------
with glass_card("glass-inputs"):
    section_header("edit_note", "Patient Snapshot")
    col1, col2 = st.columns(2)
    ui_values = {}

    with col1:
        for key in ["age", "gender", "icu_los", "hr", "temp", "resp"]:
            spec = mu.CURATED_INPUTS[key]
            if spec["widget"] == "radio":
                ui_values[key] = st.radio(spec["label"], list(spec["options"].keys()),
                                           horizontal=True, help=spec["help"])
            else:
                ui_values[key] = st.slider(spec["label"], spec["min"], spec["max"], spec["default"],
                                            step=spec.get("step", 1), help=spec["help"])

    with col2:
        for key in ["o2sat", "sbp", "map", "wbc", "lactate"]:
            spec = mu.CURATED_INPUTS[key]
            ui_values[key] = st.slider(spec["label"], spec["min"], spec["max"], spec["default"],
                                        step=spec.get("step", 1), help=spec["help"])

    predict_clicked = st.button("Estimate Sepsis Risk", type="primary", width="stretch",
                                 icon=":material/monitor_heart:")

# -----------------------------------------------------------------------
# Run prediction
# -----------------------------------------------------------------------
if predict_clicked:
    st.session_state.pop("chat_messages", None)

    feature_values = mu.user_inputs_to_feature_values(ui_values)
    with st.spinner(f"Running {n_samples} background predictions with {available[model_key]}..."):
        probs = mu.predict_risk_distribution(
            model_bundle, feature_values, feature_cols, distributions,
            n_samples=n_samples, base_seed=None
        )
    mean_prob = float(probs.mean())
    band, color = mu.risk_band(mean_prob)

    st.session_state["last_result"] = {
        "mean_prob": mean_prob, "band": band, "color": color, "probs": probs,
        "n_samples": n_samples, "model_key": model_key,
        "curated_labels": {mu.CURATED_INPUTS[k]["label"]: v for k, v in ui_values.items()},
    }

    try:
        db.save_prediction(
            user_email=auth.current_user()["email"],
            curated_inputs=st.session_state["last_result"]["curated_labels"],
            mean_prob=mean_prob, risk_band=band,
            prob_min=float(probs.min()), prob_max=float(probs.max()),
            n_samples=n_samples, model_name=available[model_key],
        )
    except Exception as e:
        st.warning(f"Prediction complete, but couldn't save to history: {e}")

# -----------------------------------------------------------------------
# Show result
# -----------------------------------------------------------------------
if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    mean_prob, band, color = result["mean_prob"], result["band"], result["color"]
    probs, n_samples_used = result["probs"], result["n_samples"]
    curated_labels = result["curated_labels"]
    model_label = available[result["model_key"]]

    with glass_card("glass-result"):
        section_header("assessment", "Result")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Average predicted risk", f"{mean_prob:.1%}")
        r2.metric("Risk band", band)
        r3.metric("Range across draws", f"{probs.min():.1%} – {probs.max():.1%}")
        r4.metric("Model used", model_label)

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=mean_prob * 100, number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]}, 'bar': {'color': color},
                'steps': [
                    {'range': [0, 20], 'color': '#123322'},
                    {'range': [20, 50], 'color': '#3a3016'},
                    {'range': [50, 100], 'color': '#3a1616'},
                ],
            },
            title={'text': "Sepsis Risk"},
        ))
        fig.update_layout(
            height=300, margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#F2F3F5"),
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown(
            f"**{band} risk** ({mean_prob:.1%} average across {n_samples_used} draws, "
            f"using **{model_label}**). This reflects your entered values plus randomly sampled "
            "background features consistent with the training population — treat it as illustrative, not diagnostic."
        )

        with st.expander("See distribution across all draws", icon=":material/bar_chart:"):
            hist_fig = go.Figure(go.Histogram(x=probs * 100, nbinsx=20, marker_color=color))
            hist_fig.update_layout(
                xaxis_title="Predicted risk (%)", yaxis_title="Count of draws",
                height=280, margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
                font=dict(color="#F2F3F5"),
            )
            st.plotly_chart(hist_fig, width="stretch")

    # -------------------------------------------------------------------
    # AI explanation chat (Groq / gpt-oss-120b)
    # -------------------------------------------------------------------
    with glass_card("glass-chat"):
        section_header("forum", "Ask AI to explain this result")

        if "chat_messages" not in st.session_state:
            if st.button("Explain this result using AI", icon=":material/auto_awesome:"):
                st.session_state["chat_messages"] = [
                    {"role": "system", "content": llm.SYSTEM_PROMPT},
                    {"role": "user", "content": llm.build_initial_prompt(mean_prob, band, curated_labels)},
                ]
                with st.spinner("Asking the AI..."):
                    try:
                        reply = llm.call_groq(st.session_state["chat_messages"])
                        st.session_state["chat_messages"].append({"role": "assistant", "content": reply})
                    except Exception as e:
                        st.error(f"Couldn't reach the AI explainer: {e}")
                        st.session_state.pop("chat_messages", None)
                st.rerun()

        if "chat_messages" in st.session_state:
            for msg in st.session_state["chat_messages"][1:]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            follow_up = st.chat_input("Ask a follow-up question...")
            if follow_up:
                st.session_state["chat_messages"].append({"role": "user", "content": follow_up})
                with st.spinner("Thinking..."):
                    try:
                        reply = llm.call_groq(st.session_state["chat_messages"])
                        st.session_state["chat_messages"].append({"role": "assistant", "content": reply})
                    except Exception as e:
                        st.error(f"Couldn't reach the AI explainer: {e}")
                st.rerun()

else:
    st.info("Set the values above and click **Estimate Sepsis Risk**.", icon=":material/arrow_upward:")
