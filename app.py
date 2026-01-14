import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import os
import plotly.graph_objects as go

# =============================================================================
# 1. VISUAL CONFIGURATION (CSS HACKING)
# =============================================================================
st.set_page_config(page_title="Perovskite TE Predictor", layout="wide")

# This CSS blocks makes Streamlit look EXACTLY like your HTML image
st.markdown("""
<style>
    /* 1. GLOBAL FONT */
    html, body, [class*="css"] {
        font-family: 'Arial', sans-serif;
        background-color: #F8F9FA; /* Light grey background */
    }

    /* 2. CENTER TITLE */
    h1 {
        text-align: center;
        color: #172B4D;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0px;
        padding-bottom: 10px;
    }

    /* 3. CENTER & STYLE INPUTS */
    /* This targets the input box to make it centered and grey */
    div[data-testid="stTextInput"] {
        text-align: center;
        margin: 0 auto;
    }
    div[data-testid="stTextInput"] input {
        text-align: center;
        font-weight: bold;
        font-size: 18px;
        background-color: #E9ECEF;
        color: #333;
        border-radius: 8px;
        border: 1px solid #CED4DA;
        padding: 10px;
    }

    /* 4. STYLE BUTTON (RED/ORANGE) */
    div.stButton > button {
        width: 100%;
        background-color: #ff4b4b; /* The Red/Orange color */
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 12px 20px;
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    div.stButton > button:hover {
        background-color: #ff3333;
        color: white;
        border: none;
    }

    /* 5. HIDE DEFAULT STREAMLIT MENU */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 6. STATUS BAR AT BOTTOM */
    .status-box {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #ffffff;
        padding: 10px;
        text-align: center;
        border-top: 1px solid #ddd;
        font-weight: bold;
        color: #333;
        z-index: 9999;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. BACKEND LOGIC (UNCHANGED)
# =============================================================================
# --- PICKLE PATCH ---
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

# --- SETUP ---
BASE_MODEL_DIR = "final_models"
PROPERTIES_DB_PATH = "data/elemental_properties.xlsx"
PROP_MAP = {"Z":"Atomic_Number", "IE":"Ionization_Energy_kJ_per_mol", "EN":"Electronegativity_Pauling", "EA":"Electron_Affinity_kJ_per_mol", "IR":"Ionic_Radius_pm", "MP":"Melting_Point_C", "BP":"Boiling_Point_C", "AD":"Atomic_Density_g_per_cm3", "HoE":"Heat_of_Evaporation_kJ_per_mol", "HoF":"Heat_of_Fusion_kJ_per_mol"}
A_SITE = {"Ca","Sr","Ba","Pb","La","Nd","Sm","Gd","Dy","Ho","Eu","Pr","Na","K","Ce","Bi","Er","Yb","Cu","Y","In","Sb"}
B_SITE = {"Ti","Zr","Nb","Co","Mn","Fe","W","Sn","Hf","Ni","Ta","Ir","Mo","Ru","Rh","Cr"}
X_SITE = {"O"}

MODELS_CONFIG = {
    "S": {"file": "Seebeck_Coefficient_S_μV_K__ExtraTrees.pkl", "name": "Seebeck Coefficient", "unit": "μV/K", "color": "#1f77b4"},
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
    
    # Strict checks
    if abs(sum(A.values()) - 1.0) > 0.05: raise ValueError(f"A-site sum is {sum(A.values()):.2f}, must be 1.0")
    if abs(sum(B.values()) - 1.0) > 0.05: raise ValueError(f"B-site sum is {sum(B.values()):.2f}, must be 1.0")
    return A, B

def prepare_input(model, A, B, T, elem_props):
    req = model.get_feature_names()
    N = len(T)
    vals = {}
    for p, col in PROP_MAP.items():
        vals[f"{p}_A"] = sum(elem_props[e][col] * r for e, r in A.items())
        vals[f"{p}_B"] = sum(elem_props[e][col] * r for e, r in B.items())
    tf = (vals["IR_A"] + 140.0) / (1.414 * (vals["IR_B"] + 140.0))
    vals["Tf"], vals["τ"] = tf, tf
    data = {col: (T if col == "T" else np.full(N, vals.get(col, 0))) for col in req}
    return pd.DataFrame(data), tf

# =============================================================================
# 3. UI LAYOUT (MATCHING IMAGE 2)
# =============================================================================

# A. Header
st.markdown("<h1>Perovskite TE Predictor</h1>", unsafe_allow_html=True)

# B. Input & Button (Centered using columns)
# We use [1, 2, 1] to push the input to the center, just like your image
col_left, col_mid, col_right = st.columns([1, 1.5, 1])

with col_mid:
    # 1. Formula Input
    formula = st.text_input("Formula", value="La0.2Ca0.8TiO3", label_visibility="collapsed")
    
    # 2. Red Button (Full Width of col_mid)
    btn = st.button("Analyze Composition")

# C. Logic & Plotting
elem_props = load_resources()
models = load_models()
status_msg = "System Ready | Waiting for input..."

if btn and elem_props:
    try:
        A, B = parse_formula(formula.strip())
        temps = np.arange(300, 1101, 50)
        
        # Create Grid for Plots (2x2)
        c1, c2 = st.columns(2)
        c3, c4 = st.columns(2)
        grid_locs = [c1, c2, c3, c4]
        
        tf_val = 0
        idx = 0
        
        for key in ["S", "Sigma", "Kappa", "zT"]:
            if key in models:
                cfg = MODELS_CONFIG[key]
                X, tf_val = prepare_input(models[key], A, B, temps, elem_props)
                preds = models[key].predict(X)
                
                # PLOTLY SETUP (Clean, Lines+Markers, White BG)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=temps, y=preds,
                    mode='lines+markers',
                    line=dict(width=4, color=cfg['color']),
                    marker=dict(size=8, color=cfg['color']),
                    name=cfg['name']
                ))
                
                fig.update_layout(
                    title=dict(text=cfg['name'], x=0.5, font=dict(size=18, color="#333", family="Arial, sans-serif")),
                    xaxis=dict(title="Temperature (K)", showgrid=True, gridcolor='#F0F0F0', zeroline=False),
                    yaxis=dict(title=cfg['unit'], showgrid=True, gridcolor='#F0F0F0', zeroline=False),
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    margin=dict(l=40, r=20, t=50, b=40),
                    height=300
                )
                
                with grid_locs[idx]:
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                idx += 1

        status_msg = f"Tolerance Factor: {tf_val:.3f} | A-Site Sum: 1.00 | B-Site Sum: 1.00"

    except Exception as e:
        status_msg = f"Error: {str(e)}"
        st.error(str(e))

# D. Sticky Status Bar
st.markdown(f"""
<div class="status-box">
    {status_msg}
</div>
""", unsafe_allow_html=True)
