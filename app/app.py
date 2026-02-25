import streamlit as st
import joblib
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load model ────────────────────────────────────────────────────────────────
model = joblib.load("model/diabetes_pipeline.pkl")

MODEL_METRICS = {"Recall": 81.5, "AUC": 82.61}
THRESHOLD = 0.45

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ---- Google Font ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---- Reduce Streamlit default padding ---- */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2rem !important;
    }

    /* ---- Background ---- */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f2027 100%);
        min-height: 100vh;
    }

    /* ---- Hide default Streamlit chrome ---- */
    #MainMenu, footer, header { visibility: hidden; }

    /* ---- Hero banner ---- */
    .hero {
        background: linear-gradient(120deg, #06b6d4 0%, #3b82f6 50%, #8b5cf6 100%);
        border-radius: 16px;
        padding: 20px 32px;
        margin-bottom: 20px;
        box-shadow: 0 12px 36px rgba(6,182,212,0.25);
        position: relative;
        overflow: hidden;
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -60%; right: -5%;
        width: 220px; height: 220px;
        background: rgba(255,255,255,0.06);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero::after {
        content: '';
        position: absolute;
        bottom: -50%; left: 3%;
        width: 140px; height: 140px;
        background: rgba(255,255,255,0.04);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-emoji {
        font-size: 2.8rem;
        line-height: 1;
        flex-shrink: 0;
        filter: drop-shadow(0 2px 8px rgba(0,0,0,0.3));
    }
    .hero-text h1 {
        font-size: 1.55rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 4px 0;
        letter-spacing: -0.3px;
    }
    .hero-text p {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.82);
        margin: 0;
        line-height: 1.5;
    }

    /* ---- Section label ---- */
    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #06b6d4;
        margin: 0 0 12px 0;
    }

    /* ---- Input card ---- */
    .input-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.09);
        border-top: 2px solid rgba(6,182,212,0.35);
        border-radius: 14px;
        padding: 16px 18px 12px;
        backdrop-filter: blur(10px);
        transition: border-color 0.25s, transform 0.2s;
    }
    .input-card:hover {
        border-color: rgba(6,182,212,0.5);
        transform: translateY(-2px);
    }
    .input-icon {
        font-size: 1.3rem;
        margin-bottom: 4px;
    }
    .input-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #94a3b8;
        margin-bottom: 2px;
        letter-spacing: 0.3px;
    }
    .input-hint {
        font-size: 0.7rem;
        color: #475569;
        margin-top: 2px;
    }

    /* ---- Streamlit number_input styling ---- */
    .stNumberInput > div > div > input {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        padding: 8px 12px !important;
    }
    .stNumberInput > div > div > input:focus {
        border-color: #06b6d4 !important;
        box-shadow: 0 0 0 3px rgba(6,182,212,0.18) !important;
    }
    .stNumberInput label { color: #cbd5e1 !important; font-weight: 500 !important; }

    /* ---- Predict button ---- */
    .stButton > button {
        background: linear-gradient(120deg, #06b6d4, #3b82f6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 13px 40px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 24px rgba(59,130,246,0.3) !important;
        cursor: pointer !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 32px rgba(59,130,246,0.45) !important;
    }
    .stButton > button:active { transform: translateY(0) !important; }

    /* ---- Result cards ---- */
    .result-high {
        background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(220,38,38,0.06));
        border: 1px solid rgba(239,68,68,0.35);
        border-top: 3px solid #ef4444;
        border-radius: 16px;
        padding: 18px 28px;
        text-align: center;
        animation: fadeInUp 0.45s ease;
    }
    .result-low {
        background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(5,150,105,0.06));
        border: 1px solid rgba(16,185,129,0.35);
        border-top: 3px solid #10b981;
        border-radius: 16px;
        padding: 18px 28px;
        text-align: center;
        animation: fadeInUp 0.45s ease;
    }
    .result-icon { font-size: 2.2rem; margin-bottom: 6px; }
    .result-label {
        font-size: 1.3rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 5px;
    }
    .result-high .result-label { color: #f87171; }
    .result-low  .result-label { color: #34d399; }
    .result-desc {
        font-size: 0.87rem;
        color: #94a3b8;
        line-height: 1.5;
        max-width: 380px;
        margin: 0 auto;
    }

    /* ── Risk bar ── */
    .risk-bar-bg {
        background: rgba(255,255,255,0.07);
        border-radius: 50px;
        height: 10px;
        margin: 18px 0 5px;
        overflow: hidden;
    }
    .risk-bar-fill-high {
        height: 100%;
        border-radius: 50px;
        background: linear-gradient(90deg, #f59e0b, #ef4444);
    }
    .risk-bar-fill-low {
        height: 100%;
        border-radius: 50px;
        background: linear-gradient(90deg, #06b6d4, #10b981);
    }
    .risk-pct { font-size: 0.75rem; color: #64748b; text-align: right; }

    /* ---- Metric pill ---- */
    .metric-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 8px; }
    .metric-pill {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 50px;
        padding: 6px 16px;
        font-size: 0.8rem;
        color: #cbd5e1;
    }
    .metric-pill span { color: #06b6d4; font-weight: 700; }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: rgba(15,23,42,0.97) !important;
        border-right: 1px solid rgba(255,255,255,0.07) !important;
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #06b6d4 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1.2px !important;
        margin-bottom: 0 !important;
    }

    /* ---- Tip card ---- */
    .tip-card {
        background: rgba(255,255,255,0.03);
        border-left: 3px solid #06b6d4;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.82rem;
        color: #94a3b8;
        line-height: 1.5;
    }

    /* ---- Divider ---- */
    .fancy-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(6,182,212,0.35), transparent);
        margin: 22px 0;
        border: none;
    }

    /* ---- BMI badge ---- */
    .bmi-badge {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 50px;
        padding: 9px 20px;
        font-size: 0.88rem;
        color: #94a3b8;
    }
    .bmi-badge .bmi-val { font-size: 1.1rem; font-weight: 700; color: #06b6d4; }
    .bmi-badge .bmi-cat { font-size: 0.75rem; font-weight: 600; padding: 3px 10px; border-radius: 50px; }
    .cat-under  { background: rgba(59,130,246,0.2);  color: #60a5fa; }
    .cat-normal { background: rgba(16,185,129,0.2);  color: #34d399; }
    .cat-over   { background: rgba(245,158,11,0.2);  color: #fbbf24; }
    .cat-obese  { background: rgba(239,68,68,0.2);   color: #f87171; }

    /* ---- BMI mode toggle (radio as pill switcher) ---- */
    div[data-testid="stRadio"] > label { display: none !important; }
    div[data-testid="stRadio"] > div {
        flex-direction: row !important;
        gap: 0 !important;
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 50px !important;
        padding: 4px !important;
        width: fit-content !important;
        margin: 0 auto !important;
    }
    div[data-testid="stRadio"] > div > label {
        border-radius: 50px !important;
        padding: 5px 14px !important;
        margin: 0 !important;
        cursor: pointer !important;
        color: #64748b !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
    }
    div[data-testid="stRadio"] > div > label:has(input:checked) {
        background: linear-gradient(120deg, #06b6d4, #3b82f6) !important;
        color: #ffffff !important;
    }
    div[data-testid="stRadio"] > div > label > div:first-child { display: none !important; }

    /* ---- Social link buttons ---- */
    .social-links { display: flex; flex-direction: column; gap: 7px; margin-top: 10px; }
    .social-btn {
        display: flex;
        align-items: center;
        gap: 10px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 10px;
        padding: 9px 13px;
        text-decoration: none !important;
        color: #cbd5e1 !important;
        font-size: 0.83rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    .social-btn:hover {
        background: rgba(6,182,212,0.1);
        border-color: rgba(6,182,212,0.4);
        color: #06b6d4 !important;
        transform: translateX(3px);
    }
    .social-btn .s-icon { font-size: 1rem; }
    .social-btn .s-label { flex: 1; }
    .social-btn .s-arrow { color: #475569; font-size: 0.72rem; }

    /* ---- Dataset info cards ---- */
    .info-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin-bottom: 22px;
    }
    .sidebar-info-grid {
        display: grid;
        grid-template-columns: 1fr;
        gap: 8px;
        margin-top: 10px;
    }
    .info-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 10px;
        padding: 8px 12px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .info-card:nth-child(1) { border-top: 2px solid #06b6d4; }
    .info-card:nth-child(2) { border-top: 2px solid #8b5cf6; }
    .info-card:nth-child(3) { border-top: 2px solid #3b82f6; }
    .info-card:nth-child(4) { border-top: 2px solid #10b981; }
    .info-card-icon { font-size: 1.1rem; flex-shrink: 0; }
    .info-card-body { flex: 1; min-width: 0; }
    .info-card-label {
        font-size: 0.62rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 1px;
    }
    .info-card-value { font-size: 0.8rem; font-weight: 600; color: #e2e8f0; line-height: 1.3; }
    .info-card-sub   { font-size: 0.69rem; color: #64748b; margin-top: 1px; line-height: 1.2; }

    /* ---- Animations ---- */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0);    }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 About the Dataset")
    st.markdown(
        """
        <div class="sidebar-info-grid">
            <div class="info-card">
                <div class="info-card-icon">🏛️</div>
                <div class="info-card-body">
                    <div class="info-card-label">Data Source</div>
                    <div class="info-card-value">NIDDK</div>
                    <div class="info-card-sub">National Institute of Diabetes</div>
                </div>
            </div>
            <div class="info-card">
                <div class="info-card-icon">👩</div>
                <div class="info-card-body">
                    <div class="info-card-label">Population</div>
                    <div class="info-card-value">Pima Indian Heritage</div>
                    <div class="info-card-sub">Females aged ≥ 21 yrs</div>
                </div>
            </div>
            <div class="info-card">
                <div class="info-card-icon">🎯</div>
                <div class="info-card-body">
                    <div class="info-card-label">Objective</div>
                    <div class="info-card-value">Binary Classification</div>
                    <div class="info-card-sub">Diabetic vs Non-diabetic</div>
                </div>
            </div>
            <div class="info-card">
                <div class="info-card-icon">🔬</div>
                <div class="info-card-body">
                    <div class="info-card-label">Features</div>
                    <div class="info-card-value">Glucose · BMI · Age · Pregnancies</div>
                    <div class="info-card-sub">By correlation analysis</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    st.markdown("### 👨‍💻 Created By")
    st.markdown(
        """
        <p style="color:#e2e8f0;font-size:0.95rem;font-weight:600;margin-bottom:12px">
            Osama Abd El-Mohsen
        </p>
        <div class="social-links">
            <a class="social-btn" href="https://github.com/Osama-Abd-El-Mohsen" target="_blank">
                <span class="s-icon">🐙</span>
                <span class="s-label">GitHub</span>
                <span class="s-arrow">↗</span>
            </a>
            <a class="social-btn" href="https://osama-abd-elmohsen-portfolio.me/" target="_blank">
                <span class="s-icon">🌐</span>
                <span class="s-label">Portfolio</span>
                <span class="s-arrow">↗</span>
            </a>
            <a class="social-btn" href="https://www.linkedin.com/in/osama-abd-el-mohsen" target="_blank">
                <span class="s-icon">💼</span>
                <span class="s-label">LinkedIn</span>
                <span class="s-arrow">↗</span>
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    st.markdown("### 📊 Model Performance")
    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-pill">Recall&nbsp;<span>{MODEL_METRICS['Recall']}%</span></div>
            <div class="metric-pill">AUC&nbsp;<span>{MODEL_METRICS['AUC']}%</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#475569;font-size:0.75rem;text-align:center">'
        "For educational purposes only.<br>Not a substitute for medical advice.</p>",
        unsafe_allow_html=True,
    )

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <div class="hero-emoji">🩺</div>
        <div class="hero-text">
            <h1>Diabetes Risk Predictor</h1>
            <p>Enter your clinical measurements below and get an instant AI-powered
            assessment of your diabetes risk level. Results are indicative —
            always consult a healthcare professional.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Input fields ──────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Clinical Inputs</p>', unsafe_allow_html=True)

# BMI mode toggle
_, toggle_col, _ = st.columns([1, 4, 1])
with toggle_col:
    bmi_mode = st.radio(
        "BMI input mode",
        options=["⚖️ Enter Weight & Height", "🔢 Enter BMI directly"],
        horizontal=True,
        label_visibility="collapsed",
    )

st.markdown("<br>", unsafe_allow_html=True)

use_wh = bmi_mode == "⚖️ Enter Weight & Height"

if use_wh:
    col1, col2, col3, col4, col5 = st.columns(5, gap="medium")
else:
    col1, col2, col3, col4 = st.columns(4, gap="medium")

# ── Glucose (always col1) ──
with col1:
    st.markdown(
        '<div class="input-card">'
        '<div class="input-icon">🩸</div>'
        '<div class="input-title">Glucose Level</div>'
        '<div class="input-hint">mg/dL — fasting plasma glucose</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    glucose = st.number_input(
        "Glucose Level (mg/dL)",
        min_value=0.0, max_value=300.0, value=100.0, step=1.0,
        label_visibility="collapsed",
    )

# ── BMI columns (mode-dependent) ──
if use_wh:
    with col2:
        st.markdown(
            '<div class="input-card">'
            '<div class="input-icon">⚖️</div>'
            '<div class="input-title">Weight</div>'
            '<div class="input-hint">kilograms (kg)</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        weight_kg = st.number_input(
            "Weight (kg)",
            min_value=1.0, max_value=300.0, value=70.0, step=0.5,
            label_visibility="collapsed",
        )

    with col3:
        st.markdown(
            '<div class="input-card">'
            '<div class="input-icon">📏</div>'
            '<div class="input-title">Height</div>'
            '<div class="input-hint">centimeters (cm)</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        height_cm = st.number_input(
            "Height (cm)",
            min_value=50.0, max_value=250.0, value=170.0, step=0.5,
            label_visibility="collapsed",
        )

    age_col, preg_col = col4, col5

else:
    with col2:
        st.markdown(
            '<div class="input-card">'
            '<div class="input-icon">📊</div>'
            '<div class="input-title">BMI</div>'
            '<div class="input-hint">Body Mass Index (kg/m²)</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        bmi_direct = st.number_input(
            "BMI",
            min_value=10.0, max_value=70.0, value=25.0, step=0.1,
            label_visibility="collapsed",
        )

    age_col, preg_col = col3, col4

# ── Age ──
with age_col:
    st.markdown(
        '<div class="input-card">'
        '<div class="input-icon">🎂</div>'
        '<div class="input-title">Age</div>'
        '<div class="input-hint">Years — your current age</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    age = st.number_input(
        "Age",
        min_value=0, max_value=120, value=30, step=1,
        label_visibility="collapsed",
    )

# ── Pregnancies ──
with preg_col:
    st.markdown(
        '<div class="input-card">'
        '<div class="input-icon">🤰</div>'
        '<div class="input-title">Pregnancies</div>'
        '<div class="input-hint">Number of times pregnant</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0, max_value=20, value=0, step=1,
        label_visibility="collapsed",
    )

# ── Resolve final BMI value ────────────────────────────────────────────────────
if use_wh:
    height_m = height_cm / 100.0
    bmi = round(weight_kg / (height_m ** 2), 1)
else:
    bmi = round(bmi_direct, 1)

if bmi < 18.5:
    cat_label, cat_class = "Underweight", "cat-under"
elif bmi < 25.0:
    cat_label, cat_class = "Normal", "cat-normal"
elif bmi < 30.0:
    cat_label, cat_class = "Overweight", "cat-over"
else:
    cat_label, cat_class = "Obese", "cat-obese"

st.markdown("<br>", unsafe_allow_html=True)
_, bmi_col, _ = st.columns([2, 3, 2])
with bmi_col:
    label = "Calculated BMI" if use_wh else "BMI"
    st.markdown(
        f"""
        <div style="text-align:center">
            <div class="bmi-badge">
                <span>{label}</span>
                <span class="bmi-val">{bmi}</span>
                <span class="bmi-cat {cat_class}">{cat_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict button ─────────────────────────────────────────────────────────────
_, btn_col, _ = st.columns([2, 3, 2])
with btn_col:
    predict_clicked = st.button("🔍  Analyze My Risk", use_container_width=True)

st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

# ── Result ─────────────────────────────────────────────────────────────────────
if predict_clicked:
    data = np.array([[glucose, bmi, age, pregnancies]])
    proba = float(model.predict_proba(data)[:, 1][0])
    is_high = proba >= THRESHOLD
    pct = int(round(proba * 100))

    _, res_col, _ = st.columns([1, 4, 1])
    with res_col:
        if is_high:
            bar_class = "risk-bar-fill-high"
            st.markdown(
                f'<div class="result-high">'
                f'<div class="result-icon">⚠️</div>'
                f'<div class="result-label">High Risk Detected</div>'
                f'<div class="result-desc">Our model estimates a <strong style="color:#f87171">{pct}% probability</strong> of diabetes based on your inputs. Please consult a healthcare professional for a proper diagnosis.</div>'
                f'<div class="risk-bar-bg"><div class="{bar_class}" style="width:{pct}%"></div></div>'
                f'<div class="risk-pct">{pct}% risk score</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            bar_class = "risk-bar-fill-low"
            st.markdown(
                f'<div class="result-low">'
                f'<div class="result-icon">✅</div>'
                f'<div class="result-label">Low Risk</div>'
                f'<div class="result-desc">Our model estimates a <strong style="color:#34d399">{pct}% probability</strong> of diabetes based on your inputs. Keep maintaining your healthy lifestyle and schedule regular check-ups.</div>'
                f'<div class="risk-bar-bg"><div class="{bar_class}" style="width:{pct}%"></div></div>'
                f'<div class="risk-pct">{pct}% risk score</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
else:
    _, placeholder_col, _ = st.columns([1, 4, 1])
    with placeholder_col:
        st.markdown(
            """
            <div style="
                border: 2px dashed rgba(255,255,255,0.1);
                border-radius: 20px;
                padding: 48px;
                text-align: center;
                color: #475569;
            ">
                <div style="font-size:3rem;margin-bottom:12px">📋</div>
                <p style="font-size:1rem;margin:0">
                    Fill in your clinical measurements above and click
                    <strong style="color:#06b6d4">Analyze My Risk</strong> to see your result.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
