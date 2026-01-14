import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import os
import plotly.graph_objects as go

# =============================================================================
# PAGE SETUP
# =============================================================================
st.set_page_config(page_title="Perovskite TE Predictor", layout="wide", initial_sidebar_state="collapsed")

# =============================================================================
# CLEAN & MODERN CSS (matching your reference image style)
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, sans-serif !important;
    }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 5rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        max-width: 1400px;
        margin: auto;
    }

    .main-title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 800;
        color: #1a1a2e;
        margin: 0.5rem 0 1.5rem 0;
        letter-spacing: -0.6px;
    }

    .input-row {
        max-width: 700px;
        margin: 0 auto 2rem auto;
    }

    .input-row input {
        border: 2px solid #cbd5e1;
        border-radius: 12px;
        font-size: 1.3rem;
        font-weight: 600;
        text-align: center;
        padding: 16px;
        height: 3.4rem;
    }

    .analyze-btn {
        background: #3b82f6 !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        height: 3.4rem !important;
        margin-left: 1rem !important;
        transition: all 0.2s;
    }

    .analyze-btn:hover {
        background: #2563eb !important;
        transform: translateY(-1px);
    }

    .plot-card {
        background: white;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        padding: 1rem;
        margin-bottom: 1.5rem;
    }

    .plot-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #111827;
        text-align: center;
        margin-bottom: 0.8rem;
    }

    .bottom-info {
        font-size: 1.1rem;
        color: #374151;
        text-align: center;
        padding: 1rem;
        background: #f9fafb;
        border-top: 1px solid #e5e7eb;
        border-radius: 0 0 12px 12px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">Perovskite TE Predictor</div>', unsafe_allow_html=True)

# Input + Button
with st.container():
    st.markdown('<div class="input-row">', unsafe_allow_html=True)
    col_input, col_btn = st.columns([5, 2])
    with col_input:
        formula = st.text_input("", value="La0.2Ca0.8TiO3", label_visibility="collapsed")
    with col_btn:
        btn = st.button("Analyze", key="analyze_btn", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# BACKEND LOGIC (all functions defined here - error fixed!)
# =============================================================================

class FeatureAwareModel:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = list(feature_names)

    def predict(self, X):
        return self.model.predict(X[self.feature_names])

    def get_feature_names(self):
        return self.feature_names

import __main__
setattr(__main__, "FeatureAwareModel", FeatureAwareModel)

BASE_MODEL_DIR = "final_models"
PROPERTIES_DB_PATH = "data/elemental_properties.xlsx"

PROP_MAP = {
    "Z": "Atomic_Number",
    "IE": "Ionization_Energy_kJ_per_mol",
    "EN": "Electronegativity_Pauling",
    "EA": "Electron_Affinity_kJ_per_mol",
    "IR": "Ionic_Radius_pm",
    "MP": "Melting_Point_C",
    "BP": "Boiling_Point_C",
    "AD": "Atomic_Density_g_per_cm3",
    "HoE": "Heat_of_Evaporation_kJ_per_mol",
    "HoF": "Heat_of_Fusion_kJ_per_mol"
}

A_SITE = {"Ca","Sr","Ba","Pb","La","Nd","Sm","Gd","Dy","Ho","Eu","Pr","Na","K","Ce","Bi","Er","Yb","Cu","Y","In","Sb"}
B_SITE = {"Ti","Zr","Nb","Co","Mn","Fe","W","Sn","Hf","Ni","Ta","Ir","Mo","Ru","Rh","Cr"}
X_SITE = {"O"}

MODELS_CONFIG = {
    "S": {"file": "Seebeck_Coefficient_S_μV_K__ExtraTrees.pkl", "name": "Seebeck Coefficient", "unit": "µV/K", "color": "#1f77b4"},
    "Sigma": {"file": "Electrical_Conductivity_σ_S_cm__CatBoost.pkl", "name": "Electrical Conductivity", "unit": "S/cm", "color": "#ff7f0e"},
    "Kappa": {"file": "Thermal_Conductivity_κ_W_m-K__GradientBoost.pkl", "name": "Thermal Conductivity", "unit": "W/m·K", "color": "#2ca02c"},
    "zT": {"file": "Figure_of_Merit_zT_CatBoost.pkl", "name": "Figure of Merit (zT)", "unit": "dimensionless", "color": "#d62728"}
}

@st.cache_data
def load_resources():
    if not os.path.exists(PROPERTIES_DB_PATH):
        return {}
    df = pd.read_excel(PROPERTIES_DB_PATH)
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    return df.set_index("Element").T.to_dict()

@st.cache_resource
def load_models():
    models = {}
    for k, cfg in MODELS_CONFIG.items():
        path = os.path.join(BASE_MODEL_DIR, cfg["file"])
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[k] = pickle.load(f)
    return models

def parse_formula(formula):
    pattern = re.compile(r"([A-Z][a-z]*)(\d*\.?\d*)")
    parts = pattern.findall(formula)
    if not parts: raise ValueError("Invalid Formula")
    elements = {}
    for el, amt in parts:
        amt = float(amt) if amt else 1.0
        elements[el] = elements.get(el, 0.0) + amt
    A, B = {}, {}
    for el, amt in elements.items():
        if el in X_SITE: continue
        elif el in A_SITE: A[el] = amt
        elif el in B_SITE: B[el] = amt
        else: raise ValueError(f"Unknown element: {el}")
    if abs(sum(A.values()) - 1.0) > 0.05: raise ValueError(f"A-site sum is {sum(A.values()):.2f}, must be 1.0")
    if abs(sum(B.values()) - 1.0) > 0.05: raise ValueError(f"B-site sum is {sum(B.values()):.2f}, must be 1.0")
    return A, B

def prepare_input(model, A, B, T, elem_props):
    req = model.get_feature_names()
    N = len(T)
    vals = {}
    for p, col in PROP_MAP.items():
        vals[f"{p}_A"] = sum(elem_props.get(e, {}).get(col, 0) * r for e, r in A.items())
        vals[f"{p}_B"] = sum(elem_props.get(e, {}).get(col, 0) * r for e, r in B.items())
    tf = (vals["IR_A"] + 140.0) / (1.414 * (vals["IR_B"] + 140.0))
    vals["Tf"], vals["τ"] = tf, tf
    data = {col: (T if col == "T" else np.full(N, vals.get(col, 0))) for col in req}
    return pd.DataFrame(data), tf, vals

# =============================================================================
# MAIN LOGIC
# =============================================================================
elem_props = load_resources()
models = load_models()

if btn:
    try:
        A, B = parse_formula(formula.strip())
        temps = np.arange(300, 1101, 50)

        row1 = st.columns(2, gap="large")
        row2 = st.columns(2, gap="large")
        cols = row1 + row2

        plot_idx = 0
        tf_val = 0  # Will be updated

        for key in ["S", "Sigma", "Kappa", "zT"]:
            if key not in models:
                continue
            cfg = MODELS_CONFIG[key]
            X, tf_val, _ = prepare_input(models[key], A, B, temps, elem_props)
            preds = models[key].predict(X)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=temps, y=preds,
                mode='lines+markers',
                line=dict(width=3, color=cfg['color']),
                marker=dict(size=7, color=cfg['color'], line=dict(width=1, color='white'))
            ))

            fig.update_layout(
                title_text=cfg['name'],
                title_x=0.5,
                title_font=dict(size=20, color="#111827"),
                height=400,
                margin=dict(l=60, r=30, t=70, b=70),
                xaxis_title="Temperature (K)",
                yaxis_title=f"{cfg['unit']}",
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(family="Inter", size=13, color="#374151"),
                xaxis=dict(showline=True, linewidth=1.6, linecolor='#4b5563', mirror=True,
                           ticks="outside", gridcolor='rgba(209,213,219,0.5)'),
                yaxis=dict(showline=True, linewidth=1.6, linecolor='#4b5563', mirror=True,
                           ticks="outside", gridcolor='rgba(209,213,219,0.5)'),
                hovermode="x unified",
                showlegend=False
            )

            with cols[plot_idx]:
                st.markdown('<div class="plot-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="plot-title">{cfg["name"]}</div>', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                st.markdown('</div>', unsafe_allow_html=True)

            plot_idx += 1

        # Bottom info (like your image)
        st.markdown(f"""
        <div class="bottom-info">
            Tolerance Factor: <b>{tf_val:.3f}</b> | 
            A-site: <b>{A}</b> | 
            B-site: <b>{B}</b>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error during calculation: {str(e)}")
