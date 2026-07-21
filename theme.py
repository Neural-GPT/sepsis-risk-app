"""
Shared visual theme for the Sepsis Risk Estimator app.

Call inject_theme() once at the top of every page.
Use `with glass_card("unique-key"):` to wrap content in a real glass panel
(this nests correctly in the DOM, unlike raw markdown div injection).
Use section_header(icon, title, subtitle) for liquid-glass icon headers.
Use render_sidebar(user) for the shared, minimal sidebar nav.
"""
import streamlit as st

BG = "#08090C"
GLASS_BG = "rgba(255, 255, 255, 0.045)"
GLASS_BORDER = "rgba(255, 255, 255, 0.09)"
GLASS_BORDER_HOVER = "rgba(255, 255, 255, 0.16)"
ACCENT = "#4FD1FF"
ACCENT_2 = "#9B7BFF"
TEXT = "#F2F3F5"
MUTED = "#9AA2AE"

MATERIAL_FONT_LINK = (
    '<link rel="stylesheet" '
    'href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:'
    'opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap" />'
)

THEME_CSS = f"""
<style>
.stApp {{
    background:
        radial-gradient(circle at 12% -10%, rgba(79,209,255,0.08) 0%, transparent 40%),
        radial-gradient(circle at 90% 0%, rgba(155,123,255,0.07) 0%, transparent 45%),
        {BG};
    color: {TEXT};
}}

/* ---- Sidebar: thin + minimal ---- */
section[data-testid="stSidebar"] {{
    width: 250px !important;
    min-width: 250px !important;
    background: rgba(10, 11, 14, 0.65);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-right: 1px solid {GLASS_BORDER};
}}
section[data-testid="stSidebar"] > div {{ width: 250px !important; padding-top: 0.5rem; }}
[data-testid="stSidebarContent"] {{ padding: 1.1rem 0.9rem; }}

/* ---- Hide default streamlit chrome ---- */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header[data-testid="stHeader"] {{ background: transparent; }}

/* ---- Glass containers (via st.container(key="glass-...")) ---- */
div[class*="st-key-glass-"] {{
    background: {GLASS_BG} !important;
    border: 1px solid {GLASS_BORDER} !important;
    border-radius: 18px !important;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    padding: 1.4rem 1.6rem !important;
    box-shadow: 0 8px 30px rgba(0,0,0,0.30);
    transition: border-color 0.2s ease;
}}
div[class*="st-key-glass-"]:hover {{ border-color: {GLASS_BORDER_HOVER} !important; }}

/* ---- Sidebar nav (page_link) ---- */
[data-testid="stSidebar"] [data-testid="stPageLink"] {{
    border-radius: 10px;
    padding: 0.35rem 0.6rem !important;
    margin-bottom: 2px;
    transition: background 0.15s ease;
}}
[data-testid="stSidebar"] [data-testid="stPageLink"]:hover {{
    background: rgba(255,255,255,0.05);
}}
[data-testid="stSidebar"] [data-testid="stPageLink"] p {{
    font-size: 0.87rem !important;
    color: {TEXT} !important;
}}
[data-testid="stIconMaterial"] {{
    color: {ACCENT} !important;
    filter: drop-shadow(0 0 5px rgba(79,209,255,0.45));
}}

/* ---- Sidebar user card ---- */
.sb-user {{ display:flex; align-items:center; gap:0.6rem; padding: 0.3rem 0.2rem 0.9rem 0.2rem; }}
.sb-avatar {{
    width: 34px; height: 34px; border-radius: 50%;
    background: linear-gradient(135deg, {ACCENT}, {ACCENT_2});
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.85rem; color: #05070A;
    box-shadow: 0 0 14px rgba(79,209,255,0.35);
    flex-shrink: 0;
}}
.sb-user-name {{ font-size: 0.85rem; font-weight: 600; color: {TEXT}; line-height:1.2; }}
.sb-user-email {{ font-size: 0.72rem; color: {MUTED}; line-height:1.2; }}
.sb-nav-label {{
    font-size: 0.68rem; letter-spacing: 1.5px; color: {MUTED};
    text-transform: uppercase; margin: 0.2rem 0 0.4rem 0.2rem;
}}
.sb-divider {{ height:1px; background:{GLASS_BORDER}; margin: 0.9rem 0; }}

/* ---- Liquid-glass icon badge + section headers ---- */
.icon-badge {{
    border-radius: 13px;
    background: linear-gradient(135deg, rgba(79,209,255,0.16), rgba(155,123,255,0.16));
    border: 1px solid rgba(255,255,255,0.14);
    backdrop-filter: blur(10px);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 18px rgba(79,209,255,0.12), inset 0 1px 0 rgba(255,255,255,0.08);
    flex-shrink: 0;
}}
.icon-badge .material-symbols-outlined {{
    font-size: 21px;
    background: linear-gradient(135deg, {ACCENT}, {ACCENT_2});
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    filter: drop-shadow(0 0 6px rgba(79,209,255,0.4));
}}
.section-header {{ display:flex; align-items:center; gap:0.8rem; margin: 0.3rem 0 1.1rem 0; }}
.section-title {{ font-size: 1.35rem; font-weight: 700; color: {TEXT}; line-height:1.2; }}
.section-subtitle {{ font-size: 0.82rem; color: {MUTED}; margin-top:2px; }}

/* ---- Headings ---- */
h1, h2, h3 {{ color: {TEXT} !important; font-weight: 700 !important; letter-spacing: -0.3px; }}

/* ---- Buttons: gradient glass pill ---- */
.stButton > button {{
    background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_2} 100%);
    color: #05070A;
    border: none;
    border-radius: 999px;
    font-weight: 600;
    padding: 0.55rem 1.4rem;
    box-shadow: 0 4px 20px rgba(79,209,255,0.25);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
.stButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 6px 26px rgba(79,209,255,0.4);
    color: #05070A;
}}
.stButton > button p {{ color: #05070A !important; font-weight: 600; }}

/* ---- Segmented control (model switcher) ---- */
[data-testid="stSegmentedControl"] label {{
    background: {GLASS_BG} !important;
    border: 1px solid {GLASS_BORDER} !important;
    backdrop-filter: blur(10px);
}}
[data-testid="stSegmentedControl"] label[data-checked="true"] {{
    background: linear-gradient(135deg, rgba(79,209,255,0.25), rgba(155,123,255,0.25)) !important;
    border-color: {ACCENT} !important;
}}

/* ---- Inputs / sliders / selects ---- */
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {{
    background: rgba(255,255,255,0.045) !important;
    border: 1px solid {GLASS_BORDER} !important;
    border-radius: 10px !important;
    color: {TEXT} !important;
}}
.stSlider [data-baseweb="slider"] div[role="slider"] {{ background-color: {ACCENT} !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
.stTabs [data-baseweb="tab"] {{
    background: {GLASS_BG}; border-radius: 10px 10px 0 0;
    border: 1px solid {GLASS_BORDER}; border-bottom: none;
}}

/* ---- Metrics ---- */
div[data-testid="stMetric"] {{
    background: rgba(255,255,255,0.03);
    border: 1px solid {GLASS_BORDER};
    border-radius: 14px;
    padding: 0.8rem 1rem;
}}
div[data-testid="stMetricLabel"] {{ color: {MUTED} !important; font-size: 0.78rem !important; }}
div[data-testid="stMetricValue"] {{ color: {ACCENT} !important; }}

/* ---- Expander / tables / caption ---- */
.streamlit-expanderHeader {{
    background: {GLASS_BG} !important; border-radius: 10px !important;
    border: 1px solid {GLASS_BORDER} !important; color: {TEXT} !important;
}}
.stCaption, small {{ color: {MUTED} !important; }}
hr {{ border-color: {GLASS_BORDER} !important; }}
</style>
"""


def inject_theme():
    st.markdown(MATERIAL_FONT_LINK, unsafe_allow_html=True)
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def glass_card(key: str):
    """Context manager returning a real, nestable glass-panel container.
    Usage: with glass_card('glass-my-section'): ... widgets ...
    `key` must start with 'glass-' and be unique on the page."""
    assert key.startswith("glass-"), "glass_card keys must start with 'glass-'"
    return st.container(key=key)


def icon_badge_html(icon_name: str, size: int = 40) -> str:
    return (
        f'<div class="icon-badge" style="width:{size}px;height:{size}px;">'
        f'<span class="material-symbols-outlined">{icon_name}</span></div>'
    )


def section_header(icon_name: str, title: str, subtitle: str = None):
    sub = f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="section-header">{icon_badge_html(icon_name)}'
        f'<div><div class="section-title">{title}</div>{sub}</div></div>',
        unsafe_allow_html=True,
    )


def render_sidebar(user=None):
    """Shared, minimal sidebar nav. Call inside `with st.sidebar:` at the
    top of each page. Returns nothing; page-specific settings/logout are
    added by the caller after this."""
    if user:
        st.markdown(
            f'<div class="sb-user">'
            f'<div class="sb-avatar">{user["name"][:1].upper()}</div>'
            f'<div><div class="sb-user-name">{user["name"]}</div>'
            f'<div class="sb-user-email">{user["email"]}</div></div></div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="sb-nav-label">Navigate</div>', unsafe_allow_html=True)
    st.page_link("app.py", label="Home", icon=":material/home:")
    st.page_link("pages/1_Predict.py", label="Predict", icon=":material/biotech:")
    st.page_link("pages/2_History.py", label="History", icon=":material/history:")
    st.page_link("pages/3_About.py", label="About & Models", icon=":material/info:")
