import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import os
import plotly.graph_objects as go

# =============================================================================
# 1. PAGE SETUP & CSS (The Design Engine)
# =============================================================================
st.set_page_config(page_title="Perovskite TE Predictor", layout="wide")

st.markdown("""
<style>
    /* --- GLOBAL RESET & BACKGROUND --- */
    .stApp {
        background-color: #F5F7FA; /* The Light Gray Background */
        font-family: Arial, Helvetica, sans-serif;
    }
    
    /* Remove default Streamlit top padding to fit the header */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem; /* Space for status bar */
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100%;
    }

    /* --- 1. HEADER STYLE --- */
    .custom-header {
        background-color: white;
        padding: 12px 0;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #172B4D;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-radius: 8px;
    }

    /* --- 2. INPUT BAR STYLE --- */
    /* Style the input text box */
    div[data-testid="stTextInput"] input {
        border: 2px solid #DFE1E6;
        border-radius: 4px;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
        padding: 8px 12px;
    }
    
    /* Style the Button to match the HTML Blue */
    div.stButton > button {
        background-color: #0052CC;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 4px;
        padding: 9px 20px;
        font-size: 16px;
        width: 100%;
        transition: background 0.2s;
    }
    div.stButton > button:hover {
        background-color: #0747a6;
        color: white;
    }

    /* --- 3. STATUS BAR (Fixed Bottom) --- */
    .status-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #E1E4E8;
        border-top: 2px solid #172B4D;
        color: #172B4D;
        text-align: center;
        padding: 10px 15px;
        font-size: 14px;
        font-weight: bold;
        z-index: 9999;
    }

    /* Hide standard Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. BACKEND LOGIC (Standard)
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
PROP_MAP = {"Z":"Atomic_Number", "IE":"Ionization_Energy_kJ_per_mol", "EN":"Electronegativity_Pauling", "EA":"Electron_Affinity_kJ_per_mol", "IR":"Ionic_Radius_pm", "MP":"Melting_Point_C", "BP":"Boiling_Point_C", "AD":"Atomic_Density_g_per_cm3", "HE":"Heat_of_Evaporation_kJ_per_mol", "HF":"Heat_of_Fusion_kJ_per_mol"}
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
        else: raise ValueError(f"Unknown: {el}")
    
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
# 3. UI LAYOUT
# =============================================================================

# --- A. HEADER ---
st.markdown('<div class="custom-header">Perovskite TE Predictor</div>', unsafe_allow_html=True)

# --- B. INPUT BAR (Centered) ---
# [Spacer, Input(3), Button(1), Spacer]
c_left, c_input, c_btn, c_right = st.columns([2, 3, 1, 2], gap="small")

with c_input:
    formula = st.text_input("Formula", value="La0.2Ca0.8TiO3", label_visibility="collapsed", placeholder="Enter Formula...")

with c_btn:
    btn = st.button("Analyze Composition")

# --- C. MAIN GRID LOGIC ---
elem_props = load_resources()
models = load_models()
status_msg = "System Ready | Waiting for input..."

if btn and elem_props:
    try:
        A, B = parse_formula(formula.strip())
        temps = np.arange(300, 1101, 50)
        
        # Create 2x2 Grid
        row1 = st.columns(2)
        row2 = st.columns(2)
        grid_locs = row1 + row2 
        
        tf_val = 0
        idx = 0
        
        for key in ["S", "Sigma", "Kappa", "zT"]:
            if key in models:
                cfg = MODELS_CONFIG[key]
                X, tf_val = prepare_input(models[key], A, B, temps, elem_props)
                preds = models[key].predict(X)
                
                # --- PLOTLY CARD CONFIGURATION ---
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=temps, y=preds,
                    mode='lines+markers',
                    line=dict(width=3, color=cfg['color']),
                    marker=dict(size=6, color=cfg['color']),
                ))
                
                fig.update_layout(
                    # Title inside the plot
                    title=dict(
                        text=f"<b>{cfg['name']}</b>", 
                        x=0.5, 
                        font=dict(size=16, color="#172B4D")
                    ),
                    # Axes Styling
                    xaxis=dict(
                        title="<b>Temperature (K)</b>", 
                        showgrid=True, gridcolor='#F0F0F0',
                        showline=True, linewidth=2, linecolor='#172B4D',
                        mirror=True, ticks="outside"
                    ),
                    yaxis=dict(
                        title=f"<b>{cfg['unit']}</b>", 
                        showgrid=True, gridcolor='#F0F0F0',
                        showline=True, linewidth=2, linecolor='#172B4D',
                        mirror=True, ticks="outside"
                    ),
                    # CARD EFFECT: White background + Margins
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    margin=dict(l=60, r=20, t=50, b=50),
                    height=300, # Fixed height for uniformity
                )
                
                with grid_locs[idx]:
                    # We wrap the chart in a div to give it the shadow/radius via CSS if needed,
                    # but Plotly handles the white background well.
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    
                idx += 1

        status_msg = f"Tolerance Factor: {tf_val:.3f} | A-site Sum: 1.00 | B-site Sum: 1.00"

    except Exception as e:
        status_msg = f"Error: {str(e)}"
        st.error(str(e))

# --- D. STATUS BAR ---
st.markdown(f'<div class="status-bar">{status_msg}</div>', unsafe_allow_html=True)
