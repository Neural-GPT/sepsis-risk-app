import streamlit as st

from theme import inject_theme, glass_card
import auth_utils as auth

st.set_page_config(
    page_title="Sepsis Risk Estimator",
    page_icon=":material/monitor_heart:",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_theme()

if auth.is_logged_in():
    st.switch_page("pages/1_Predict.py")

# -----------------------------------------------------------------------
# Hero
# -----------------------------------------------------------------------
hero_l, hero_r = st.columns([1.3, 1], gap="large")

with hero_l:
    st.markdown(
        f"<div style='margin-top:2.2rem;'>"
        f"<span style='color:{'#4FD1FF'}; letter-spacing:3px; font-weight:600; font-size:0.8rem;'>"
        f"MACHINE LEARNING &nbsp;&middot;&nbsp; CLINICAL RISK SCREENING</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h1 style='font-size:3.1rem; line-height:1.1; margin-top:0.5rem;'>"
        "Sepsis Risk<br>Estimator</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#9AA2AE; font-size:1.1rem; max-width:34rem; margin-top:0.9rem;'>"
        "An interactive, explainable machine learning demo for early sepsis risk "
        "screening — trained on the PhysioNet 2019 Challenge dataset, with a choice "
        "of XGBoost, Random Forest, or Logistic Regression, plus a built-in AI "
        "explanation layer.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#5C636E; font-size:0.82rem; margin-top:1.4rem;'>"
        "Educational demo only — not a medical device, not for clinical use.</p>",
        unsafe_allow_html=True,
    )

with hero_r:
    with glass_card("glass-auth"):
        tab_login, tab_signup = st.tabs(["Log In", "Sign Up"])

        with tab_login:
            st.markdown("#### Welcome back")
            login_email = st.text_input("Email", key="login_email")
            login_password = st.text_input("Password", type="password", key="login_password")
            if st.button("Log In", key="login_btn", width="stretch"):
                success, msg = auth.login(login_email, login_password)
                if success:
                    st.success(msg)
                    st.switch_page("pages/1_Predict.py")
                else:
                    st.error(msg)

        with tab_signup:
            st.markdown("#### Create an account")
            signup_name = st.text_input("Full name", key="signup_name")
            signup_email = st.text_input("Email", key="signup_email")
            signup_password = st.text_input("Password", type="password", key="signup_password",
                                             help="At least 8 characters.")
            signup_confirm = st.text_input("Confirm password", type="password", key="signup_confirm")
            if st.button("Create Account", key="signup_btn", width="stretch"):
                success, msg = auth.signup(signup_email, signup_name, signup_password, signup_confirm)
                if success:
                    st.success(msg)
                    st.switch_page("pages/1_Predict.py")
                else:
                    st.error(msg)

st.write("")
st.write("")

# -----------------------------------------------------------------------
# Feature strip
# -----------------------------------------------------------------------
f1, f2, f3, f4 = st.columns(4, gap="medium")
features = [
    ("psychology", "Multiple Models", "Switch between XGBoost, Random Forest, and Logistic Regression."),
    ("monitoring", "Rich Visualizations", "Risk gauges, distribution histograms, and model comparison charts."),
    ("forum", "AI Explanations", "GPT-OSS-120B explains each result in plain English."),
    ("database", "Prediction History", "Every prediction you run is saved to your account."),
]
for col, (icon, title, desc) in zip([f1, f2, f3, f4], features):
    with col:
        with glass_card(f"glass-feat-{icon}"):
            st.markdown(
                f'<div class="icon-badge" style="width:42px;height:42px;">'
                f'<span class="material-symbols-outlined">{icon}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**{title}**")
            st.caption(desc)
