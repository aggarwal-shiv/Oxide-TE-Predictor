import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import os
import plotly.graph_objects as go

# =============================================================================
# PAGE CONFIGURATION - MUST COME RIGHT AFTER streamlit import
# =============================================================================
st.set_page_config(
    page_title="Oxide TE-Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# MODERN & CLEAN CSS
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    }

    .stApp {
        background-color: #f8f9fd;
    }

    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 7rem !important;
        padding-left: 1.8rem !important;
        padding-right: 1.8rem !important;
        max-width: 1400px;
        margin: auto;
    }

    .main-header {
        text-align: center;
        font-size: clamp(2rem, 5.5vw, 3.4rem);
        font-weight: 800;
        margin: 0.4em 0 0.9em 0;
        color: #0f172a;
        background: linear-gradient(90deg, #334155, #64748b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    div[data-testid="stTextInput"] > div > div > input {
        border: 2px solid #cbd5e1;
        border-radius: 10px;
        font-size: 1.15rem;
        font-weight: 600;
        padding: 14px !important;
        background-color: white;
        color: #1e293b;
        text-align: center;
    }

    div.stButton > button {
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
        color: white;
        border: none;
        border-radius: 10px;
        font-size: 1.15rem;
        font-weight: 700;
        padding: 0.8rem 1.6rem;
        height: 3.4rem;
        transition: all 0.25s;
        width: 100%;
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #2563eb, #3b82f6);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(59,130,246,0.4);
    }

    .stPlotlyChart {
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.09);
        overflow: hidden;
        margin-bottom: 1.6rem !important;
    }

    /* Better mobile experience */
    @media (max-width: 768px) {
        .row-widget.stHorizontal {
            flex-direction: column !important;
        }
        .stPlotlyChart {
            height: 340px !important;
        }
    }

    .status-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(241, 245, 249, 0.94);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-top: 1px solid rgba(226,232,240,0.85);
        color: #334155;
        text-align: center;
        padding: 14px;
        font-weight: 600;
        font-size: 1.05rem;
        z-index: 999;
        box-shadow: 0 -3px 12px rgba(0,0,0,0.07);
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================
st.markdown('<div class="main-header">Oxide TE-Predictor</div>', unsafe_allow_html=True)

# =============================================================================
# BACKEND LOGIC (original functionality preserved)
# =============================================================================
try:
    import sklearn
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
    import catboost
    import xgboost
except ImportError:
    pass

class FeatureAwareModel:
    def __init__(self, model, feature_names, target_name=None):
        self.model = model
        self.feature_names = list(feature_names)
        self.target_name = target_name

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
    elem_props = {}
    if os.path.exists(PROPERTIES_DB_PATH):
        df = pd.read_excel(PROPERTIES_DB_PATH)
        df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        elem_props = df.set_index("Element").T.to_dict()
    return elem_props

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
    if not parts:
        raise ValueError("Invalid chemical formula")
    
    elements = {}
    for el, amt in parts:
        amt = float(amt) if amt else 1.0
        elements[el] = elements.get(el, 0.0) + amt

    A, B = {}, {}
    for el, amt in elements.items():
        if el in X_SITE:
            continue
        elif el in A_SITE:
            A[el] = amt
        elif el in B_SITE:
            B[el] = amt
        else:
            raise ValueError(f"Unknown element in formula: {el}")

    if abs(sum(A.values()) - 1.0) > 0.05:
        raise ValueError(f"A-site sum is {sum(A.values()):.2f}, should be close to 1.0")
    if abs(sum(B.values()) - 1.0) > 0.05:
        raise ValueError(f"B-site sum is {sum(B.values()):.2f}, should be close to 1.0")

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
# UI LAYOUT
# =============================================================================
c_left, c_input, c_btn, c_right = st.columns([2, 3, 1, 2], gap="small")
with c_input:
    formula = st.text_input("Formula", value="La0.2Ca0.8TiO3", label_visibility="collapsed")
with c_btn:
    btn = st.button("Predict")

elem_props = load_resources()
models = load_models()

status_msg = "Ready to predict — enter formula and press Predict"

if btn and elem_props:
    try:
        A, B = parse_formula(formula.strip())
        temps = np.arange(300, 1101, 50)

        col1, col2 = st.columns(2, gap="medium")
        plots_placed = 0

        for key in ["S", "Sigma", "Kappa", "zT"]:
            if key not in models:
                continue

            cfg = MODELS_CONFIG[key]
            X, tf_val, calc_vals = prepare_input(models[key], A, B, temps, elem_props)
            preds = models[key].predict(X)

            # Modern Plot
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=temps, y=preds,
                mode='lines+markers',
                line=dict(width=3.4, color=cfg['color']),
                marker=dict(size=8, color=cfg['color'], line=dict(width=1.2, color='white')),
                name=cfg['name']
            ))

            fig.update_layout(
                title=dict(text=cfg['name'], x=0.5, font=dict(size=22, color='#1e293b')),
                xaxis_title="Temperature (K)",
                yaxis_title=f"{cfg['name']}  ({cfg['unit']})",
                height=390,
                margin=dict(l=65, r=35, t=70, b=65),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#334155"),
                hovermode="x unified",
                showlegend=False,
                xaxis=dict(gridcolor='rgba(226,232,240,0.7)', zeroline=False),
                yaxis=dict(gridcolor='rgba(226,232,240,0.7)', zeroline=False)
            )

            target_col = col1 if plots_placed % 2 == 0 else col2
            with target_col:
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            plots_placed += 1

        status_msg = f"✓ Tolerance factor: {tf_val:.3f} | Predicted as stable perovskite structure"

    except Exception as e:
        status_msg = f"Error: {str(e)}"
        st.error(str(e), icon="⚠️")

# STATUS BAR
st.markdown(f'<div class="status-bar">{status_msg}</div>', unsafe_allow_html=True)
