"""FoodLens — AI food detection and nutrition analysis.

    streamlit run app.py
"""
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import torch
from PIL import Image

# ── Path setup ───────────────────────────────────────────────────────────────
_SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(_SRC))

from predict import load_model                                  # noqa: E402
from pipeline import run_pipeline, CLASSIFIER_CONF_THRESHOLD   # noqa: E402
# Palette, chart builders and box drawing are shared with server.py via charts.py.
from charts import (                                            # noqa: E402
    BLUE, NAVY, SLATE, BORDER, BG, GREEN, AMBER, RED, BOX_COLORS,
    annotate_image, make_macro_pie, make_calorie_bar,
    load_history, training_plot,
)

# ── Paths ────────────────────────────────────────────────────────────────────
CHECKPOINT    = Path("models/classifier_best.pt")
HISTORY_CSV   = Path("outputs/history.csv")
CONFUSION_PNG = Path("outputs/figures/confusion_matrix.png")
DAILY_GOAL    = 2000

st.set_page_config(
    page_title="FoodLens",
    page_icon="🍽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── BASE & FONT ──────────────────────────────────────────────── */
*, html, body {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    box-sizing: border-box;
}

[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded' !important;
    font-feature-settings: 'liga' !important;
    -webkit-font-feature-settings: 'liga' !important;
    -moz-font-feature-settings: 'liga' !important;
    -webkit-font-smoothing: antialiased !important;
    font-style: normal !important;
    font-weight: 400 !important;
    text-transform: none !important;
    white-space: nowrap !important;
    letter-spacing: normal !important;
}

/* ── CHROME REMOVAL ──────────────────────────────────────────── */
#MainMenu                              { display: none !important; }
footer                                 { display: none !important; }
.stDeployButton                        { display: none !important; }
[data-testid="stStatusWidget"]         { display: none !important; }
[data-testid="stDecoration"]           { display: none !important; }
[data-testid="stHeaderActionElements"] { display: none !important; }

header[data-testid="stHeader"] {
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
    overflow: visible !important;
}

[data-testid="stToolbar"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    position: static !important;
    height: 0 !important;
    min-height: 0 !important;
    overflow: visible !important;
}

/* Sidebar collapse control — fixed position so it lines up with the logo row. */
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"] {
    position: fixed !important;
    top: 20px !important;
    left: 194px !important;
    z-index: 999999 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 30px !important;
    height: 30px !important;
    padding: 0 !important;
    margin: 0 !important;
    background: rgba(255,255,255,0.08) !important;
    border-radius: 6px !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    transition: background 0.15s !important;
}
[data-testid="stExpandSidebarButton"]:hover,
[data-testid="stSidebarCollapseButton"]:hover {
    background: rgba(255,255,255,0.16) !important;
}
[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"],
[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"] {
    color: #94A3B8 !important;
    font-size: 19px !important;
}

/* ── TOP-SPACE FIX ───────────────────────────────────────────── */
.main .block-container,
[data-testid="stMainBlockContainer"] {
    padding-top: 0 !important;
    padding-bottom: 2.5rem !important;
    max-width: 1300px;
}

/* ── APP BACKGROUND ──────────────────────────────────────────── */
.stApp,
[data-testid="stAppViewContainer"],
.main { background: #F8FAFC !important; }

/* Streamlit spaces every sidebar element; zeroing that lets the logo sit at the
   sidebar's own padding-top. */
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"],
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"],
section[data-testid="stSidebar"] [data-testid="stElementContainer"],
section[data-testid="stSidebar"] .element-container {
    gap: 0 !important;
    margin: 0 !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    margin: 0 !important;
}

/* ── SIDEBAR: DARK NAVY ───────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #0F172A !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    min-width: 230px !important;
    max-width: 240px !important;
}
section[data-testid="stSidebar"] > div:first-child {
    background: #0F172A !important;
    padding: 20px 14px 16px !important;
    display: flex !important;
    flex-direction: column !important;
    min-height: 100vh !important;
}

/* ── SIDEBAR TEXT: force all text to light colors ─────────────── */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] li {
    color: #94A3B8 !important;
}
section[data-testid="stSidebar"] .sb-app-name  { color: #F1F5F9 !important; }
section[data-testid="stSidebar"] .sb-logo-tile { color: white   !important; }
section[data-testid="stSidebar"] .sb-nav-label { color: #64748B !important; }
section[data-testid="stSidebar"] .sb-avatar    { color: #94A3B8 !important; }

.sb-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    padding-bottom: 24px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 20px;
}
.sb-logo-tile {
    width: 34px; height: 34px;
    background: #2563EB;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; font-weight: 800; color: white;
    flex-shrink: 0; letter-spacing: -0.5px;
    box-shadow: 0 2px 8px rgba(37,99,235,0.40);
}
.sb-app-name {
    font-size: 16px; font-weight: 700; color: white; letter-spacing: -0.3px;
}

.sb-nav-label {
    font-size: 10px; font-weight: 600; color: #64748B;
    text-transform: uppercase; letter-spacing: 1.3px;
    margin: 0 0 8px 4px; line-height: 1;
}

.sb-spacer { flex: 1 1 0; min-height: 32px; }

.sb-avatar-row {
    display: flex; align-items: center; gap: 10px;
    padding: 12px 8px; margin-top: 4px;
    border-top: 1px solid rgba(255,255,255,0.07);
}
.sb-avatar {
    width: 30px; height: 30px;
    background: #1E293B; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 700; color: #94A3B8; flex-shrink: 0;
}
.sb-avatar-label { font-size: 13px; color: #64748B; font-weight: 500; }

/* ── SIDEBAR RADIO → NAV ITEMS ───────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stRadio"] > label:first-of-type {
    display: none !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: column !important;
    gap: 3px !important;
    width: 100% !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    display: flex !important;
    align-items: center !important;
    padding: 10px 12px !important;
    border-radius: 8px !important;
    color: #94A3B8 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: background 0.15s, color 0.15s !important;
    width: 100% !important;
    margin: 0 !important;
    line-height: 1.4 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(255,255,255,0.06) !important;
    color: #CBD5E1 !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input[type="radio"]:checked),
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input[type="radio"]:checked) span,
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input[type="radio"]:checked) div {
    background: #2563EB !important;
    color: white !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
    position: absolute !important;
    opacity: 0 !important;
    width: 0 !important;
    height: 0 !important;
    pointer-events: none !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label > div {
    display: contents !important;
}

/* ── PAGE HEADER BAR ─────────────────────────────────────────── */
.page-header {
    background: white;
    border-bottom: 1px solid #E2E8F0;
    padding: 0 2px 18px 0;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
}
.page-title {
    font-size: 22px;
    font-weight: 700;
    color: #0F172A;
    margin: 0;
    letter-spacing: -0.4px;
    line-height: 1;
}

/* ── SUMMARY METRIC ROW (4 cards) ───────────────────────────── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 28px;
}
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 16px 20px 14px;
    border: 1px solid #E2E8F0;
    border-left-width: 4px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.metric-card-label {
    font-size: 11px;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin: 0 0 8px;
}
.metric-card-value {
    font-size: 28px;
    font-weight: 800;
    color: #0F172A;
    line-height: 1;
    letter-spacing: -0.8px;
    margin: 0;
}
.metric-card-unit {
    font-size: 14px;
    font-weight: 600;
    color: #64748B;
}

/* ── SECTION HEADINGS ───────────────────────────────────────── */
.section-heading {
    font-size: 18px;
    font-weight: 700;
    color: #0F172A;
    margin: 0 0 14px;
    letter-spacing: -0.3px;
}

/* ── FOOD RESULT CARDS ──────────────────────────────────────── */
.food-card {
    background: white;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    border-left: 4px solid #16A34A;
    padding: 16px 18px 14px;
    margin-bottom: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.food-card-unrecog {
    border-left-color: #DC2626;
    background: #FEFEFE;
}
.food-card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 11px;
}
.food-card-name {
    font-size: 16px; font-weight: 700; color: #0F172A;
    margin: 0; letter-spacing: -0.2px; line-height: 1.3;
}
.conf-badge {
    display: inline-flex; align-items: center;
    padding: 3px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 700;
    white-space: nowrap; flex-shrink: 0;
    letter-spacing: 0.1px;
}
.conf-high { background: #DCFCE7; color: #15803D; }
.conf-med  { background: #FEF3C7; color: #B45309; }
.conf-low  { background: #FEE2E2; color: #B91C1C; }

.food-macros {
    display: flex; gap: 18px; margin-bottom: 10px; flex-wrap: wrap;
}
.food-macro-item { font-size: 14px; color: #334155; font-weight: 400; }
.food-macro-item span { font-weight: 700; color: #0F172A; }
.food-source {
    font-size: 11px; color: #94A3B8; font-style: italic; margin: 0;
}
.unrecog-note {
    font-size: 13px; color: #64748B; margin: 6px 0 0; line-height: 1.5;
}

/* ── UPLOAD ZONE ─────────────────────────────────────────────── */
[data-testid="stFileUploader"] > div {
    border: none !important;
    padding: 0 !important;
    background: transparent !important;
}
[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #CBD5E1 !important;
    border-radius: 12px !important;
    background: #F8FAFC !important;
    padding: 36px 24px !important;
    text-align: center !important;
    transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #2563EB !important;
    background: #EFF6FF !important;
}
[data-testid="stFileUploaderDropzone"] small {
    font-size: 13px !important; color: #64748B !important;
}
[data-testid="stFileUploaderDropzone"] > div > span {
    color: #64748B !important;
    font-size: 14px !important;
}
[data-testid="stFileUploader"] [data-testid="stWidgetLabel"][style*="visibility: hidden"],
[data-testid="stFileUploader"] [data-testid="stWidgetLabel"][style*="display: none"] {
    display: none !important;
}

.filename-strip {
    background: #EFF6FF; border: 1px solid #BFDBFE;
    border-radius: 8px; padding: 9px 14px; margin: 12px 0;
    font-size: 13px; color: #1D4ED8; font-weight: 500;
    display: flex; align-items: center; gap: 8px;
}

/* ── PRIMARY BUTTON ──────────────────────────────────────────── */
[data-testid="stButton"] > button[kind="primary"],
[data-testid="stBaseButton-primary"] {
    background: #2563EB !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    padding: 12px 24px !important;
    letter-spacing: 0.1px !important;
    transition: background 0.15s !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.30) !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    background: #1D4ED8 !important;
}

/* ── GOAL CARD ───────────────────────────────────────────────── */
.goal-card {
    background: white; border-radius: 12px;
    border: 1px solid #E2E8F0; padding: 20px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.goal-header {
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 10px;
}
.goal-label { font-size: 15px; font-weight: 700; color: #0F172A; margin: 0; }
.goal-pct   { font-size: 22px; font-weight: 800; color: #2563EB; }
.goal-track {
    background: #E2E8F0; border-radius: 6px;
    height: 10px; overflow: hidden; margin-bottom: 8px;
}
.goal-fill { height: 100%; background: #2563EB; border-radius: 6px; }
.goal-over .goal-fill { background: #DC2626; }
.goal-sub { display: flex; justify-content: space-between; font-size: 12px; color: #64748B; }
.goal-sub strong { color: #0F172A; }

/* ── EMPTY STATE ─────────────────────────────────────────────── */
.empty-state {
    background: white; border-radius: 12px; border: 1px solid #E2E8F0;
    text-align: center; padding: 64px 48px 56px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.empty-icon { font-size: 44px; margin-bottom: 16px; line-height: 1; }
.empty-title { font-size: 20px; font-weight: 700; color: #0F172A; margin: 0 0 10px; }
.empty-body  {
    font-size: 14px; color: #64748B; max-width: 400px;
    margin: 0 auto; line-height: 1.7;
}

/* ── SELECTBOX ───────────────────────────────────────────────── */
[data-testid="stSelectbox"] label {
    font-size: 14px !important; font-weight: 600 !important; color: #0F172A !important;
}

/* ── CHART WRAP ──────────────────────────────────────────────── */
.chart-wrap {
    background: white; border-radius: 12px;
    border: 1px solid #E2E8F0; padding: 4px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04); overflow: hidden;
}

/* ── MISC ────────────────────────────────────────────────────── */
.stAlert { border-radius: 10px !important; }
[data-testid="stCaptionContainer"] p { font-size: 12px !important; color: #94A3B8 !important; }
[data-testid="stSpinner"] > div { color: #2563EB !important; }
</style>
""", unsafe_allow_html=True)


# ── Cached model loader ───────────────────────────────────────────────────────
@st.cache_resource
def load_classifier():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes = load_model(str(CHECKPOINT), device)
    return model, classes, device


# ── Card renderers ────────────────────────────────────────────────────────────
def _conf_class(conf):
    if conf >= 0.70:
        return "conf-high", f"✓ {conf*100:.0f}% High", GREEN
    return "conf-med", f"◑ {conf*100:.0f}% Medium", AMBER


def food_card(entry):
    n     = entry["nutrition"]
    name  = entry["name"].replace("_", " ").title()
    c_class, c_label, border = _conf_class(entry["conf"])
    kcal = n["calories"] or 0
    pro  = n["protein"]  or 0
    carb = n["carbs"]    or 0
    fat  = n["fat"]      or 0
    src  = "USDA API" if n["source"] == "api" else "local fallback"
    matched = (n["matched"] or "—")[:55]
    return f"""
<div class="food-card" style="border-left-color:{border}">
  <div class="food-card-header">
    <p class="food-card-name">{name}</p>
    <span class="conf-badge {c_class}">{c_label}</span>
  </div>
  <div class="food-macros">
    <div class="food-macro-item"><span>{kcal:.0f}</span> kcal</div>
    <div class="food-macro-item"><span>{pro:.1f}g</span> protein</div>
    <div class="food-macro-item"><span>{carb:.1f}g</span> carbs</div>
    <div class="food-macro-item"><span>{fat:.1f}g</span> fat</div>
  </div>
  <p class="food-source">{src} &nbsp;&middot;&nbsp; {matched}</p>
</div>"""


def unrecognized_card(u):
    return f"""
<div class="food-card food-card-unrecog">
  <div class="food-card-header">
    <p class="food-card-name">Unrecognized food</p>
    <span class="conf-badge conf-low">&#9888; {u['conf']*100:.0f}% Below threshold</span>
  </div>
  <p class="unrecog-note">
    Outside the 20 trained classes &nbsp;&middot;&nbsp;
    top guess: &ldquo;{u['top_guess'].replace('_', ' ')}&rdquo;
  </p>
</div>"""


def summary_metrics(kcal, pro, carb, fat):
    return f"""
<div class="metric-grid">
  <div class="metric-card" style="border-left-color:{BLUE}">
    <p class="metric-card-label">Total calories</p>
    <p class="metric-card-value">{kcal:.0f}<span class="metric-card-unit"> kcal</span></p>
  </div>
  <div class="metric-card" style="border-left-color:{GREEN}">
    <p class="metric-card-label">Protein</p>
    <p class="metric-card-value">{pro:.1f}<span class="metric-card-unit">g</span></p>
  </div>
  <div class="metric-card" style="border-left-color:{AMBER}">
    <p class="metric-card-label">Carbohydrates</p>
    <p class="metric-card-value">{carb:.1f}<span class="metric-card-unit">g</span></p>
  </div>
  <div class="metric-card" style="border-left-color:{RED}">
    <p class="metric-card-label">Fat</p>
    <p class="metric-card-value">{fat:.1f}<span class="metric-card-unit">g</span></p>
  </div>
</div>"""


# ── Session state ─────────────────────────────────────────────────────────────
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "analysis_pil" not in st.session_state:
    st.session_state.analysis_pil = None
if "analysis_file_key" not in st.session_state:
    st.session_state.analysis_file_key = None


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div class="sb-logo">
  <div class="sb-logo-tile">F</div>
  <span class="sb-app-name">FoodLens</span>
</div>
<p class="sb-nav-label">Navigation</p>
""", unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["🍽  Food Analysis", "📈  Model Training"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="sb-spacer"></div>', unsafe_allow_html=True)

    st.markdown("""
<div class="sb-avatar-row">
  <div class="sb-avatar">U</div>
  <span class="sb-avatar-label">User</span>
</div>
""", unsafe_allow_html=True)


# ── Page: Food Analysis ──────────────────────────────────────────────────────
if "Food Analysis" in page:

    st.markdown('<div class="page-header"><p class="page-title">Food Analysis</p></div>',
                unsafe_allow_html=True)

    if not CHECKPOINT.exists():
        st.error(f"Classifier checkpoint not found at `{CHECKPOINT}`. "
                 "Run `src/train_classifier.py` first.")
        st.stop()

    model, classes, device = load_classifier()

    uploaded = st.file_uploader(
        "Upload a food photo",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.session_state.analysis_result = None
        st.session_state.analysis_pil    = None
        st.session_state.analysis_file_key = None
        st.markdown("""
<div class="empty-state">
  <div class="empty-icon">📷</div>
  <p class="empty-title">Upload a food photo to get started</p>
  <p class="empty-body">
    FoodLens detects individual food items, classifies each one with a
    fine-tuned MobileNetV3 neural network, and retrieves calories and
    macronutrients from the USDA FoodData Central database.
  </p>
</div>""", unsafe_allow_html=True)

    else:
        # File identity key (name + size — cheap, no need to hash content)
        file_key = f"{uploaded.name}_{uploaded.size}"

        if st.session_state.analysis_file_key != file_key:
            st.session_state.analysis_result   = None
            st.session_state.analysis_pil      = None
            st.session_state.analysis_file_key = file_key

        st.markdown(f'<div class="filename-strip">📎 &nbsp;{uploaded.name}</div>',
                    unsafe_allow_html=True)

        if st.button("Analyze Photo", type="primary", use_container_width=True):
            suffix = Path(uploaded.name).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                uploaded.seek(0)
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            try:
                with st.spinner("Detecting and classifying food regions …"):
                    pil_img = Image.open(tmp_path).convert("RGB")
                    result  = run_pipeline(tmp_path, model, classes, device)
                st.session_state.analysis_result = result
                st.session_state.analysis_pil    = pil_img
            finally:
                os.unlink(tmp_path)

        # ── Results ───────────────────────────────────────────────────────────
        if st.session_state.analysis_result:
            result       = st.session_state.analysis_result
            pil_img      = st.session_state.analysis_pil
            regions      = result["regions"]
            merged       = result["merged"]
            unrecognized = result["unrecognized"]

            if not regions:
                st.warning("No food regions detected. Try a clearer photo or a closer crop.")
            else:
                # Classify per-region for annotation labels
                from predict import predict_pil as _predict_pil

                region_labels = []
                for region in regions:
                    preds = _predict_pil(model, classes, region["crop"], device, top_k=1)
                    top_label, top_conf = preds[0]
                    region_labels.append(
                        top_label.replace("_", " ") if top_conf >= CLASSIFIER_CONF_THRESHOLD else "?"
                    )
                annotated = annotate_image(pil_img, regions, region_labels)

                total_kcal = sum(e["nutrition"]["calories"] or 0 for e in merged)
                total_pro  = sum(e["nutrition"]["protein"]  or 0 for e in merged)
                total_carb = sum(e["nutrition"]["carbs"]    or 0 for e in merged)
                total_fat  = sum(e["nutrition"]["fat"]      or 0 for e in merged)

                if merged:
                    st.markdown(summary_metrics(total_kcal, total_pro, total_carb, total_fat),
                                unsafe_allow_html=True)

                st.markdown('<p class="section-heading">Detection results</p>',
                            unsafe_allow_html=True)
                col_img, col_cards = st.columns([1.1, 1], gap="large")

                with col_img:
                    st.image(annotated, use_container_width=True)
                    st.caption(
                        f"{len(regions)} region(s) detected · "
                        f"{len(merged)} recognised · "
                        f"{len(unrecognized)} unrecognised"
                    )

                with col_cards:
                    for entry in merged:
                        st.markdown(food_card(entry), unsafe_allow_html=True)
                    for u in unrecognized:
                        st.markdown(unrecognized_card(u), unsafe_allow_html=True)

                if merged:
                    st.markdown('<p class="section-heading">Meal charts</p>',
                                unsafe_allow_html=True)
                    col_pie, col_bar = st.columns(2, gap="large")
                    with col_pie:
                        fig_pie = make_macro_pie(total_pro, total_carb, total_fat)
                        if fig_pie:
                            st.pyplot(fig_pie, use_container_width=True)
                            plt.close(fig_pie)
                    with col_bar:
                        fig_bar = make_calorie_bar(merged)
                        if fig_bar:
                            st.pyplot(fig_bar, use_container_width=True)
                            plt.close(fig_bar)

                if merged:
                    st.markdown('<p class="section-heading">Daily calorie goal</p>',
                                unsafe_allow_html=True)
                    goal_col, _ = st.columns([2, 1])
                    with goal_col:
                        pct_of_day = min(total_kcal / DAILY_GOAL, 1.0)
                        remaining  = max(DAILY_GOAL - total_kcal, 0)
                        pct_disp   = int(pct_of_day * 100)
                        over_cls   = "goal-over" if total_kcal > DAILY_GOAL else ""
                        st.markdown(f"""
<div class="goal-card {over_cls}">
  <div class="goal-header">
    <p class="goal-label">Progress toward {DAILY_GOAL} kcal daily goal</p>
    <span class="goal-pct">{pct_disp}%</span>
  </div>
  <div class="goal-track">
    <div class="goal-fill" style="width:{pct_disp}%"></div>
  </div>
  <div class="goal-sub">
    <span><strong>{total_kcal:.0f} kcal</strong> consumed</span>
    <span><strong>{remaining:.0f} kcal</strong> remaining</span>
  </div>
</div>""", unsafe_allow_html=True)


# ── Page: Model Training Graphs ──────────────────────────────────────────────
elif "Model Training" in page:

    st.markdown('<div class="page-header"><p class="page-title">Model Training Graphs</p></div>',
                unsafe_allow_html=True)

    if not HISTORY_CSV.exists():
        st.warning(f"`{HISTORY_CSV}` not found — run training first.")
    else:
        runs = load_history(HISTORY_CSV)
        run_names = [label for label, _ in runs]
        chosen_name = st.selectbox(
            "Training run",
            run_names,
            index=len(runs) - 1,
        )
        chosen_df = dict(runs)[chosen_name]

        st.markdown('<p class="section-heading">Loss and accuracy curves</p>',
                    unsafe_allow_html=True)
        col_loss, col_acc = st.columns(2, gap="large")

        with col_loss:
            fig_loss = training_plot(chosen_df, "train_loss", "val_loss",
                                     "Cross-entropy loss", "Loss")
            st.pyplot(fig_loss, use_container_width=True)
            plt.close(fig_loss)

        with col_acc:
            fig_acc = training_plot(chosen_df, "train_acc", "val_acc",
                                    "Accuracy", "Accuracy (%)", pct=True)
            st.pyplot(fig_acc, use_container_width=True)
            plt.close(fig_acc)

        st.markdown(
            "_Shaded bands: green = head phase (backbone frozen), "
            "amber = full fine-tune (all layers). "
            "Dashed = validation; solid = training._"
        )

    if CONFUSION_PNG.exists():
        st.markdown('<p class="section-heading">Confusion matrix — 20-class run</p>',
                    unsafe_allow_html=True)
        conf_img = Image.open(CONFUSION_PNG)
        st.image(conf_img, use_container_width=True)
        st.caption(
            "20×20 heatmap. Diagonal = correct predictions. "
            "Main off-diagonal pairs: fried_rice ↔ paella, tacos ↔ falafel."
        )
    elif HISTORY_CSV.exists():
        st.info(f"Confusion matrix not found at `{CONFUSION_PNG}`.")