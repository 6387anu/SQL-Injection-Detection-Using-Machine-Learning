import streamlit as st
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.sparse import issparse
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SQLi Detection",
    page_icon="🛡️",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Load models (cached so they don't reload on every rerun/interaction)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    model = joblib.load('xgb_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    explainer = shap.TreeExplainer(model)
    return model, vectorizer, explainer

model, vectorizer, explainer = load_models()

if "scan_count" not in st.session_state:
    st.session_state.scan_count = 0

# ---------------------------------------------------------------------------
# Theme / CSS
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg:            #07080d;
    --panel:         #10121b;
    --panel-alt:     #0c0e15;
    --panel-raised:  #151827;
    --hairline:      #21253866;
    --hairline-solid:#242942;
    --text:          #eef0f7;
    --text-dim:      #8b90a8;
    --text-faint:    #545a72;

    --primary:       #7c5cff;   /* electric indigo — signature accent */
    --primary-soft:  rgba(124, 92, 255, 0.12);
    --primary-border:rgba(124, 92, 255, 0.4);
    --cyan:          #22d3ee;

    --danger:        #ff4d6a;
    --danger-soft:   rgba(255, 77, 106, 0.10);
    --danger-border: rgba(255, 77, 106, 0.35);

    --safe:          #2ee6a6;
    --safe-soft:     rgba(46, 230, 166, 0.10);
    --safe-border:   rgba(46, 230, 166, 0.35);

    --radius: 14px;
    --font-display: 'Space Grotesk', sans-serif;
    --font-body: 'Inter', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}

/* ---- App backdrop: cinematic SOC-room scene (glows + light arches + node graph) ---- */
.stApp {
    background:
        radial-gradient(ellipse 900px 700px at 6% -6%, rgba(124,92,255,0.30), transparent 55%),
        radial-gradient(ellipse 700px 600px at 100% 88%, rgba(34,211,238,0.16), transparent 55%),
        radial-gradient(ellipse 500px 500px at 100% 20%, rgba(34,211,238,0.08), transparent 60%),
        linear-gradient(180deg, #0a0b14 0%, var(--bg) 40%, #030408 100%);
}

/* faint constellation / network-node motif, like a threat-map overlay, fading into the top of the page */
.stApp::before {
    content: "";
    position: fixed; inset: 0; z-index: 0; opacity: 0.4; pointer-events: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='320' height='220' viewBox='0 0 320 220'%3E%3Cg fill='none' stroke='%237c5cff' stroke-width='0.6' opacity='0.55'%3E%3Cpath d='M40 30 L130 65 L215 20 L275 75 M130 65 L170 140 M215 20 L245 115 M40 30 L75 150 M170 140 L245 115 M75 150 L170 140'/%3E%3C/g%3E%3Cg fill='%2322d3ee'%3E%3Ccircle cx='40' cy='30' r='2.1'/%3E%3Ccircle cx='130' cy='65' r='2.1'/%3E%3Ccircle cx='215' cy='20' r='2.1'/%3E%3Ccircle cx='275' cy='75' r='2.1'/%3E%3Ccircle cx='170' cy='140' r='2.1'/%3E%3Ccircle cx='245' cy='115' r='2.1'/%3E%3Ccircle cx='75' cy='150' r='2.1'/%3E%3C/g%3E%3C/svg%3E");
    background-size: 320px 220px;
    background-repeat: repeat;
    -webkit-mask-image: radial-gradient(ellipse 900px 560px at 15% 0%, black 0%, transparent 70%);
    mask-image: radial-gradient(ellipse 900px 560px at 15% 0%, black 0%, transparent 70%);
}

/* distant glowing light arches receding on the right, echoing a server-aisle */
.stApp::after {
    content: "";
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image:
        radial-gradient(ellipse 45px 640px at 68% 42%, rgba(124,92,255,0.10), transparent 70%),
        radial-gradient(ellipse 38px 560px at 78% 58%, rgba(34,211,238,0.10), transparent 70%),
        radial-gradient(ellipse 42px 600px at 88% 38%, rgba(124,92,255,0.09), transparent 70%),
        radial-gradient(ellipse 34px 520px at 96% 60%, rgba(34,211,238,0.08), transparent 70%);
    -webkit-mask-image: linear-gradient(180deg, transparent 0%, black 20%, black 80%, transparent 100%);
    mask-image: linear-gradient(180deg, transparent 0%, black 20%, black 80%, transparent 100%);
}

/* small sparkle accent, echoing the highlight in the reference scene */
.scene-sparkle {
    position: fixed; bottom: 8%; right: 6%; z-index: 0; opacity: 0.55; pointer-events: none;
    animation: twinkle 3.2s ease-in-out infinite;
}
.scene-sparkle svg { width: 22px; height: 22px; color: #cfd6ff; }
@keyframes twinkle {
    0%, 100% { opacity: 0.25; transform: scale(0.9); }
    50%      { opacity: 0.65; transform: scale(1.05); }
}

#MainMenu,header[data-testid="stHeader"], footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
header[data-testid="stHeader"] [data-testid="stToolbarActions"] { visibility: visible; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 900px; }
html, body, [class*="css"] { font-family: var(--font-body); color: var(--text); }

/* ---- Top bar ---- */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 18px; margin-bottom: 22px;
    background: rgba(16, 18, 27, 0.7);
    border: 1px solid var(--hairline-solid);
    border-radius: 999px;
    backdrop-filter: blur(14px);
}
.topbar-brand {
    display: flex; align-items: center; gap: 9px;
    font-family: var(--font-display); font-weight: 700; font-size: 14.5px; letter-spacing: 0.01em;
}
.topbar-brand svg { width: 18px; height: 18px; color: var(--primary); }
.topbar-status {
    display: flex; align-items: center; gap: 8px;
    font-family: var(--font-mono); font-size: 11.5px; color: var(--text-dim);
}
.pulse-dot {
    width: 7px; height: 7px; border-radius: 50%; background: var(--safe);
    box-shadow: 0 0 0 0 rgba(46,230,166, 0.6);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(46,230,166,0.55); }
    70%  { box-shadow: 0 0 0 7px rgba(46,230,166,0); }
    100% { box-shadow: 0 0 0 0 rgba(46,230,166,0); }
}

/* ---- Hero ---- */
.hero {
    background: linear-gradient(180deg, rgba(21,24,39,0.75), rgba(16,18,27,0.75));
    border: 1px solid var(--hairline-solid);
    border-radius: var(--radius);
    padding: 30px 32px 26px;
    backdrop-filter: blur(16px);
    margin-bottom: 18px;
}
.eyebrow {
    font-family: var(--font-mono); font-size: 11.5px; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--primary);
    display: flex; align-items: center; gap: 9px; margin-bottom: 12px;
}
.eyebrow::before {
    content: ""; width: 7px; height: 7px; border-radius: 50%;
    background: var(--primary); box-shadow: 0 0 0 4px var(--primary-soft);
}
.hero h1 {
    font-family: var(--font-display); font-size: 2.1rem; font-weight: 700;
    letter-spacing: -0.02em; margin: 0 0 8px 0;
    background: linear-gradient(90deg, #ffffff, #c9c3ff 90%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p {
    color: var(--text-dim); font-size: 14.5px; margin: 0 0 20px; max-width: 620px; line-height: 1.55;
}

.stat-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 18px; }
.stat-chip {
    display: flex; flex-direction: column; gap: 2px;
    background: var(--panel-raised); border: 1px solid var(--hairline-solid);
    border-radius: 10px; padding: 9px 14px; flex: 1; min-width: 140px;
}
.stat-chip .label {
    font-family: var(--font-mono); font-size: 9.5px; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--text-faint);
}
.stat-chip .value { font-family: var(--font-display); font-size: 14px; font-weight: 600; color: var(--text); }

.section-label {
    font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--text-faint);
    display: flex; align-items: center; gap: 10px; margin: 6px 0 10px;
}
.section-label::after { content: ""; flex: 1; height: 1px; background: var(--hairline-solid); }

/* ---- Native widget restyle ---- */
div[data-testid="stTextArea"] textarea {
    background: var(--panel-alt) !important;
    border: 1px solid var(--hairline-solid) !important;
    border-radius: 10px !important;
    font-family: var(--font-mono) !important;
    font-size: 14.5px !important;
    color: var(--text) !important;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: var(--primary-border) !important;
    box-shadow: 0 0 0 3px var(--primary-soft) !important;
}
div[data-testid="stButton"] button {
    background: linear-gradient(90deg, var(--primary), #6a4bff) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: var(--font-display) !important;
    font-weight: 600 !important;
    font-size: 16.5px !important;
    width: 100%;
    padding: 1rem 0 !important;
    box-shadow: 0 10px 24px rgba(124,92,255,0.28);
    transition: transform 0.12s ease, box-shadow 0.12s ease;
}
div[data-testid="stButton"] button:hover {
    transform: translateY(-1px);
    box-shadow: 0 14px 30px rgba(124,92,255,0.38);
}
div[data-testid="stButton"] button:active { transform: translateY(0); }

/* ---- Verdict card ---- */
.verdict-card {
    display: grid; grid-template-columns: auto 1fr auto; align-items: center; gap: 22px;
    background: var(--panel); border: 1px solid var(--hairline-solid);
    border-radius: var(--radius); padding: 24px 28px; margin: 20px 0 16px; position: relative; overflow: hidden;
}
.verdict-card.danger { border-color: var(--danger-border); background: linear-gradient(135deg, var(--danger-soft), var(--panel) 60%); }
.verdict-card.safe   { border-color: var(--safe-border);   background: linear-gradient(135deg, var(--safe-soft), var(--panel) 60%); }

.verdict-icon {
    width: 46px; height: 46px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.verdict-card.danger .verdict-icon { background: var(--danger-soft); color: var(--danger); }
.verdict-card.safe .verdict-icon   { background: var(--safe-soft);   color: var(--safe); }
.verdict-icon svg { width: 24px; height: 24px; }

.verdict-title { font-family: var(--font-display); font-size: 19px; font-weight: 700; margin-bottom: 3px; }
.verdict-card.danger .verdict-title { color: var(--danger); }
.verdict-card.safe .verdict-title   { color: var(--safe); }
.verdict-meta { font-family: var(--font-mono); font-size: 12px; color: var(--text-dim); }

.gauge-wrap { position: relative; width: 92px; height: 92px; }
.gauge-wrap svg { width: 92px; height: 92px; transform: rotate(-90deg); }
.gauge-track { fill: none; stroke: var(--hairline-solid); stroke-width: 9; }
.gauge-fill  { fill: none; stroke-width: 9; stroke-linecap: round; transition: stroke-dashoffset 0.6s ease; }
.gauge-fill.danger { stroke: var(--danger); }
.gauge-fill.safe   { stroke: var(--safe); }
.gauge-label {
    position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
    flex-direction: column; font-family: var(--font-mono);
}
.gauge-label .pct { font-size: 17px; font-weight: 600; color: var(--text); }
.gauge-label .tag { font-size: 8px; letter-spacing: 0.08em; color: var(--text-faint); text-transform: uppercase; }

/* ---- Analyst note ---- */
.explanation-box {
    background: var(--panel); border: 1px solid var(--hairline-solid); border-left: 3px solid var(--primary);
    padding: 15px 20px; border-radius: 0 12px 12px 0;
    margin: 0 0 22px; font-size: 13.5px; line-height: 1.6; color: #d3d6e6;
}
.explanation-box strong {
    font-family: var(--font-mono); font-size: 10.5px; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--primary); display: block; margin-bottom: 5px;
}

/* ---- XAI panels ---- */
.xai-box {
    background: var(--panel); border: 1px solid var(--hairline-solid); border-radius: var(--radius);
    padding: 20px 22px 6px; margin-bottom: 22px;
}
.xai-box h3 {
    font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--text-dim); margin: 0 0 16px 0; font-weight: 500;
    display: flex; align-items: center; gap: 8px;
}

/* ---- Token ledger with inline magnitude bars ---- */
.ledger-row {
    display: grid; grid-template-columns: 1fr auto 64px; align-items: center; gap: 12px;
    padding: 9px 0; border-bottom: 1px solid var(--hairline);
}
.ledger-row:last-child { border-bottom: none; }
.ledger-row code {
    font-family: var(--font-mono); background: rgba(255,255,255,0.045); padding: 3px 8px;
    border-radius: 5px; color: #dfe2ee; font-size: 12.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.ledger-row .val { font-family: var(--font-mono); font-weight: 600; font-size: 12.5px; text-align: right; }
.val-pos { color: var(--danger); }
.val-neg { color: var(--safe); }
.bar-track { height: 6px; border-radius: 4px; background: rgba(255,255,255,0.06); overflow: hidden; }
.bar-fill { height: 100%; border-radius: 4px; }
.bar-fill.val-pos { background: var(--danger); }
.bar-fill.val-neg { background: var(--safe); }

.sheet-footer {
    display: flex; justify-content: space-between; font-family: var(--font-mono);
    font-size: 11px; color: var(--text-faint); padding: 4px 4px 0;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

SHIELD_SVG = """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/></svg>"""
SHIELD_CHECK_SVG = """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/><path d="m9 12 2 2 4-4"/></svg>"""
SHIELD_ALERT_SVG = """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/><line x1="12" y1="8" x2="12" y2="13"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>"""

# ---------------------------------------------------------------------------
# Top bar + hero
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="topbar">
    <div class="topbar-brand">{SHIELD_SVG} SQLi Detection</div>
    <div class="topbar-status"><span class="pulse-dot"></span> Detection engine online</div>
</div>
<div class="scene-sparkle">
    <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 0 L14 10 L24 12 L14 14 L12 24 L10 14 L0 12 L10 10 Z"/></svg>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="eyebrow">Automated Query Inspection</div>
    <h1>Stop SQL injection before it runs</h1>
    <p>Paste a query below and the model classifies it in real time, then shows exactly which tokens drove the decision — not just a score.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<p class="section-label">Query to inspect</p>', unsafe_allow_html=True)
query_text = st.text_area(
    "Enter SQL Query here:",
    label_visibility="collapsed",
    placeholder="Paste an active query string or vector payload here...",
    height=110,
)
run = st.button("🔍  Run Inspection")

st.markdown(f"""
<div class="stat-row">
    <div class="stat-chip"><span class="label">Model</span><span class="value">XGBoost Classifier</span></div>
    <div class="stat-chip"><span class="label">Explainability</span><span class="value">SHAP TreeExplainer</span></div>
    <div class="stat-chip"><span class="label">Session scans</span><span class="value">{st.session_state.scan_count}</span></div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------
if run:
    if not query_text:
        st.warning("⚠️ Please enter a SQL query first!")
    else:
        st.session_state.scan_count += 1

        # ---- Prediction (unchanged logic) ----
        transformed_query = vectorizer.transform([query_text])
        prediction = model.predict(transformed_query)[0]
        prediction_proba = model.predict_proba(transformed_query)[0][1]

        # ---- SHAP values (unchanged logic) ----
        shap_values = explainer.shap_values(transformed_query)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        if issparse(shap_values):
            shap_values = shap_values.toarray()

        is_malicious = prediction == 1
        css_class = "danger" if is_malicious else "safe"
        confidence_pct = round(prediction_proba * 100, 1)

        # ---- Radial gauge geometry ----
        r = 40
        circumference = 2 * math.pi * r
        offset = circumference * (1 - confidence_pct / 100)

        verdict_icon = SHIELD_ALERT_SVG if is_malicious else SHIELD_CHECK_SVG
        verdict_title = "Malicious — SQL injection detected" if is_malicious else "Safe — normal query"

        st.markdown(f"""
        <div class="verdict-card {css_class}">
            <div class="verdict-icon">{verdict_icon}</div>
            <div>
                <div class="verdict-title">{verdict_title}</div>
                <div class="verdict-meta">Inspection complete · threat signature verified</div>
            </div>
            <div class="gauge-wrap">
                <svg viewBox="0 0 100 100">
                    <circle class="gauge-track" cx="50" cy="50" r="{r}"></circle>
                    <circle class="gauge-fill {css_class}" cx="50" cy="50" r="{r}"
                        stroke-dasharray="{circumference:.2f}"
                        stroke-dashoffset="{offset:.2f}"></circle>
                </svg>
                <div class="gauge-label">
                    <span class="pct">{confidence_pct}%</span>
                    <span class="tag">conf</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ---- Analyst note ----
        note = (
            "This query was flagged based on token patterns strongly associated with injection payloads "
            "in the training data. Review the ledger below for the exact n-grams driving the score."
            if is_malicious else
            "No token patterns associated with injection payloads reached a significant threshold. "
            "This query resembles benign traffic in the training data."
        )
        st.markdown(f"""
        <div class="explanation-box">
            <strong>Analyst note</strong>{note}
        </div>
        """, unsafe_allow_html=True)

        # ---- Top features ----
        feature_names = vectorizer.get_feature_names_out()
        df = pd.DataFrame({'ngram': feature_names, 'shap_value': shap_values[0]})
        df['abs_shap'] = df['shap_value'].abs()
        df_sorted = df.sort_values(by='abs_shap', ascending=False).head(10)
        max_abs = df_sorted['abs_shap'].max() or 1.0

        rows_html = ""
        for _, row in df_sorted.iterrows():
            polarity = "val-pos" if row['shap_value'] > 0 else "val-neg"
            sign = "+" if row['shap_value'] > 0 else ""
            bar_pct = round((row['abs_shap'] / max_abs) * 100, 1)
            rows_html += f"""
            <div class="ledger-row">
                <code>{row['ngram']}</code>
                <span class="val {polarity}">{sign}{row['shap_value']:.3f}</span>
                <span class="bar-track"><span class="bar-fill {polarity}" style="width:{bar_pct}%"></span></span>
            </div>"""

        col1, col2 = st.columns(2, gap="medium")

        with col1:
            st.markdown('<div class="xai-box"><h3>Feature log-odds (SHAP)</h3>', unsafe_allow_html=True)

            plt.rcParams.update({
                'figure.facecolor': '#10121b',
                'axes.facecolor': '#10121b',
                'axes.edgecolor': '#242942',
                'text.color': '#eef0f7',
                'axes.labelcolor': '#8b90a8',
                'xtick.color': '#8b90a8',
                'ytick.color': '#8b90a8',
                'font.family': 'monospace',
            })
            fig, ax = plt.subplots(figsize=(6, 4.2))
            shap.plots.bar(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=None,
                    feature_names=feature_names,
                ),
                max_display=10, show=False,
            )
            fig.patch.set_facecolor('#10121b')
            st.pyplot(fig, clear_figure=True, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="xai-box">
                <h3>Token impact ledger</h3>
                {rows_html}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sheet-footer">
            <span>Model verdicts are probabilistic and should support, not replace, analyst review.</span>
            <span>v2.0</span>
        </div>
        """, unsafe_allow_html=True)