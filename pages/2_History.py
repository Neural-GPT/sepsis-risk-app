import streamlit as st
import plotly.graph_objects as go

from theme import inject_theme, glass_card, section_header, render_sidebar
import auth_utils as auth
import db_utils as db

st.set_page_config(page_title="History — Sepsis Risk Estimator", page_icon=":material/history:", layout="wide")
inject_theme()
auth.require_login()

with st.sidebar:
    render_sidebar(auth.current_user())
    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    if st.button("Log Out", width="stretch"):
        auth.logout()
        st.switch_page("app.py")

section_header("history", "Your Prediction History", "Every prediction you've run is saved to your account.")

try:
    records = db.get_predictions_for_user(auth.current_user()["email"], limit=200)
except Exception as e:
    st.error(f"Couldn't load history: {e}")
    st.stop()

if not records:
    st.info("No predictions yet — head to the Predict page to run your first estimate.",
             icon=":material/info:")
    st.stop()

# -----------------------------------------------------------------------
# Summary stats
# -----------------------------------------------------------------------
with glass_card("glass-summary"):
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total predictions", len(records))
    avg_risk = sum(r["mean_prob"] for r in records) / len(records)
    s2.metric("Average risk (all-time)", f"{avg_risk:.1%}")
    high_count = sum(1 for r in records if r["risk_band"] == "High")
    s3.metric("High-risk results", high_count)
    models_used = len(set(r.get("model_name", "Unknown") for r in records))
    s4.metric("Models used", models_used)

# -----------------------------------------------------------------------
# Trend over time
# -----------------------------------------------------------------------
with glass_card("glass-trend"):
    section_header("trending_up", "Risk Over Time")
    records_chrono = sorted(records, key=lambda r: r["timestamp"])
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(
        x=[r["timestamp"] for r in records_chrono],
        y=[r["mean_prob"] * 100 for r in records_chrono],
        mode="lines+markers",
        line=dict(color="#4FD1FF", width=2),
        marker=dict(size=6, color="#4FD1FF"),
        name="Predicted risk (%)",
    ))
    trend_fig.update_layout(
        height=320, margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="#F2F3F5"), xaxis_title="Date", yaxis_title="Risk (%)",
    )
    trend_fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    trend_fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    st.plotly_chart(trend_fig, width="stretch")

# -----------------------------------------------------------------------
# Individual records
# -----------------------------------------------------------------------
section_header("list_alt", "All Predictions")

band_colors = {"Low": "#2e7d32", "Moderate": "#f9a825", "High": "#c62828"}

for i, r in enumerate(records):
    with glass_card(f"glass-record-{i}"):
        c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
        c1.markdown(f"**{r['timestamp'].strftime('%b %d, %Y — %H:%M UTC')}**")
        c1.caption(f"Model: {r.get('model_name', 'Unknown')}")
        c2.metric("Risk", f"{r['mean_prob']:.1%}")
        band = r["risk_band"]
        c3.markdown(
            f"<span style='background:{band_colors.get(band, '#555')}; padding:4px 12px; "
            f"border-radius:8px; font-weight:600;'>{band}</span>",
            unsafe_allow_html=True,
        )
        c4.caption(f"Range: {r['prob_min']:.1%} – {r['prob_max']:.1%} ({r['n_samples']} draws)")

        with st.expander("View inputs", icon=":material/visibility:"):
            for label, val in r["curated_inputs"].items():
                st.caption(f"**{label}:** {val}")
