import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from theme import inject_theme, glass_card, section_header, render_sidebar
import auth_utils as auth
import model_utils as mu

st.set_page_config(page_title="About & Models — Sepsis Risk Estimator", page_icon=":material/info:", layout="wide")
inject_theme()
auth.require_login()

EVAL_PATH = os.environ.get("SEPSIS_EVAL_PATH", os.path.join(os.path.dirname(__file__), "..", "model_evaluation.json"))

with st.sidebar:
    render_sidebar(auth.current_user())
    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    if st.button("Log Out", width="stretch"):
        auth.logout()
        st.switch_page("app.py")

section_header("info", "About This Project & Model Comparison",
                "Educational demo — not a medical device, not for clinical use.")

try:
    eval_data = mu.load_evaluation(EVAL_PATH)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

MODEL_ORDER = ["xgboost", "random_forest", "logistic_regression"]
MODEL_COLORS = {
    "xgboost": "#4FD1FF",
    "random_forest": "#3FE0A5",
    "logistic_regression": "#F0A93F",
}
present_models = [k for k in MODEL_ORDER if k in eval_data]

# -----------------------------------------------------------------------
# Headline metric cards, one per model
# -----------------------------------------------------------------------
section_header("bar_chart", "Model Comparison")
cols = st.columns(len(present_models))
for col, key in zip(cols, present_models):
    d = eval_data[key]
    with col:
        with glass_card(f"glass-metric-{key}"):
            st.markdown(f"**{d['model_name']}**")
            st.metric("ROC-AUC", f"{d['roc_auc']:.4f}")
            st.metric("Accuracy", f"{d['accuracy']:.1%}")
            st.metric("Avg. Precision", f"{d['average_precision']:.4f}")

st.write("")

# -----------------------------------------------------------------------
# ROC / PR curve comparison
# -----------------------------------------------------------------------
def downsample(xs, ys, max_points=300):
    if len(xs) <= max_points:
        return xs, ys
    idx = np.linspace(0, len(xs) - 1, max_points).astype(int)
    return [xs[i] for i in idx], [ys[i] for i in idx]


roc_fig = go.Figure()
for key in present_models:
    d = eval_data[key]
    fpr, tpr = downsample(d["roc_curve"]["fpr"], d["roc_curve"]["tpr"])
    roc_fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines", name=f"{d['model_name']} (AUC={d['roc_auc']:.3f})",
        line=dict(color=MODEL_COLORS.get(key, "#FFFFFF"), width=2.5),
    ))
roc_fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], mode="lines", name="Random baseline",
    line=dict(color="#666666", width=1, dash="dash"),
))
roc_fig.update_layout(
    title="ROC Curves", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
    height=430, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(color="#F2F3F5"), legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=20, r=20, t=50, b=20),
)
roc_fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
roc_fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")

pr_fig = go.Figure()
for key in present_models:
    d = eval_data[key]
    rec, prec = downsample(d["pr_curve"]["recall"], d["pr_curve"]["precision"])
    pr_fig.add_trace(go.Scatter(
        x=rec, y=prec, mode="lines", name=f"{d['model_name']} (AP={d['average_precision']:.3f})",
        line=dict(color=MODEL_COLORS.get(key, "#FFFFFF"), width=2.5),
    ))
pr_fig.update_layout(
    title="Precision-Recall Curves", xaxis_title="Recall", yaxis_title="Precision",
    height=430, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(color="#F2F3F5"), legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=20, r=20, t=50, b=20),
)
pr_fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
pr_fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")

c1, c2 = st.columns(2)
with c1:
    with glass_card("glass-roc"):
        st.plotly_chart(roc_fig, width="stretch")
with c2:
    with glass_card("glass-pr"):
        st.plotly_chart(pr_fig, width="stretch")

# -----------------------------------------------------------------------
# Per-model detail
# -----------------------------------------------------------------------
section_header("search_insights", "Per-Model Detail")

with glass_card("glass-detail"):
    tabs = st.tabs([eval_data[k]["model_name"] for k in present_models])
    for tab, key in zip(tabs, present_models):
        d = eval_data[key]
        with tab:
            left, right = st.columns([1, 1.3])

            with left:
                st.markdown("**Confusion Matrix**")
                cm = d["confusion_matrix"]
                cm_fig = go.Figure(data=go.Heatmap(
                    z=cm, x=["Pred: No Sepsis", "Pred: Sepsis"], y=["Actual: No Sepsis", "Actual: Sepsis"],
                    colorscale=[[0, "#111319"], [1, MODEL_COLORS.get(key, "#4FD1FF")]],
                    text=cm, texttemplate="%{text}", textfont=dict(size=18, color="#FFFFFF"),
                    showscale=False,
                ))
                cm_fig.update_layout(
                    height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#F2F3F5"), margin=dict(l=20, r=20, t=20, b=20),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(cm_fig, width="stretch")

            with right:
                st.markdown("**Classification Report**")
                report = d["classification_report"]
                rows = []
                for cls in ["No Sepsis", "Sepsis", "macro avg", "weighted avg"]:
                    r = report[cls]
                    rows.append({
                        "Class": cls,
                        "Precision": f"{r['precision']:.3f}",
                        "Recall": f"{r['recall']:.3f}",
                        "F1-score": f"{r['f1-score']:.3f}",
                        "Support": int(r["support"]),
                    })
                st.table(rows)
                st.caption(f"Overall accuracy: **{report['accuracy']:.1%}**")

# -----------------------------------------------------------------------
# Project info
# -----------------------------------------------------------------------
section_header("menu_book", "About This Project")
with glass_card("glass-about-text"):
    st.markdown(
        """
This application predicts sepsis risk using a machine learning pipeline trained on the
**PhysioNet/Computing in Cardiology Challenge 2019** dataset. Raw hourly ICU time-series
records were transformed into engineered per-patient features (mean, std, min, max, most
recent value, and missingness rate across 34 vitals and labs), which three different
classifiers were trained on for comparison.

Since supplying all engineered features by hand isn't practical in a demo setting, a
curated set of clinically intuitive inputs is collected directly from the user, while
every other required feature is sampled from the real training data's distribution.

**Tech stack:** XGBoost · scikit-learn · Streamlit · MongoDB · Plotly · Groq (GPT-OSS-120B)

**Live app:** [sepsis-risk-demo.onrender.com](https://sepsis-risk-demo.onrender.com/)
        """
    )
