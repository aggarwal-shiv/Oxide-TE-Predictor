import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import os
import plotly.graph_objects as go

# =============================================================================
# 1. PAGE SETUP & CSS
# =============================================================================
st.set_page_config(page_title="Oxide TE-Predictor", layout="wide")

st.markdown("""
<style>
    /* --- GLOBAL TEXT COLORS (Black) --- */
    .stApp, .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
        font-family: Arial, Helvetica, sans-serif;
    }
    
    /* --- PAGE BACKGROUND --- */
    .stApp {
        background-color: #F5F7FA;
    }
    
    /* --- MARGINS (Space Left & Right) --- */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 5rem;
        padding-left: 5rem !important;  
        padding-right: 5rem !important; 
        max-width: 100%;
    }

    /* --- HEADER --- */
    .custom-header {
        text-align: center;
        font-size: 32px;
        font-weight: 900;
        margin-bottom: 25px;
        color: #000000;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* --- INPUT BAR STYLE --- */
    div[data-testid="stTextInput"] input {
        border: 2px solid #000000;
        border-radius: 4px;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
        color: black;
        background-color: white;
    }
    
    div.stButton > button {
        background-color: #0052CC;
        color: white;
        font-weight: bold;
        border: 2px solid #0052CC;
        border-radius: 4px;
        font-size: 20px;
        height: 50px;
        width: 100%;
        margin-top: 0px; 
    }
    div.stButton > button:hover {
        background-color: #0747a6;
        border-color: #0747a6;
    }

    /* --- STATUS BAR --- */
    .status-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #E1E4E8;
        border-top: 3px solid #000000;
        color: #000000;
        text-align: center;
        padding: 12px;
        font-size: 18px;
        font-weight: bold;
        z-index: 9999;
    }

    /* Hide standard Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. BACKEND LOGIC
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
PROP_MAP = {"Z":"Atomic_Number", "IE":"Ionization_Energy_kJ_per_mol", "EN":"Electronegativity_Pauling", "EA":"Electron_Affinity_kJ_per_mol", "IR":"Ionic_Radius_pm", "MP":"Melting_Point_C", "BP":"Boiling_Point_C", "AD":"Atomic_Density_g_per_cm3", "HoE":"Heat_of_Evaporation_kJ_per_mol", "HoF":"Heat_of_Fusion_kJ_per_mol"}
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
        vals[f"{p}_A"] = sum(elem_props[e][col] * r for e, r in A.items())
        vals[f"{p}_B"] = sum(elem_props[e][col] * r for e, r in B.items())
    tf = (vals["IR_A"] + 140.0) / (1.414 * (vals["IR_B"] + 140.0))
    vals["Tf"], vals["τ"] = tf, tf
    data = {col: (T if col == "T" else np.full(N, vals.get(col, 0))) for col in req}
    
    # Return (X, tf, vals) for debugging
    return pd.DataFrame(data), tf, vals

# =============================================================================
# 3. UI LAYOUT
# =============================================================================

# --- HEADER ---
st.markdown('<div class="custom-header">Oxide TE-Predictor</div>', unsafe_allow_html=True)

# --- INPUT BAR ---
c_left, c_input, c_btn, c_right = st.columns([2, 3, 1, 2], gap="small")

with c_input:
    formula = st.text_input("Formula", value="La0.2Ca0.8TiO3", label_visibility="collapsed")

with c_btn:
    btn = st.button("Predict")

# --- MAIN GRID ---
elem_props = load_resources()
models = load_models()
status_msg = "System Ready"

if btn and elem_props:
    try:
        A, B = parse_formula(formula.strip())
        temps = np.arange(300, 1101, 50)
        
        # Grid Setup (Spacing for 16:9 look)
        row1 = st.columns(2, gap="large")
        row2 = st.columns(2, gap="large")
        grid_locs = row1 + row2 
        
        tf_val = 0
        idx = 0
        
        # --- DEBUG STORAGE FOR ALL MODELS ---
        # We will store data for every model here
        all_debug_data = {} 
        
        for key in ["S", "Sigma", "Kappa", "zT"]:
            if key in models:
                cfg = MODELS_CONFIG[key]
                X, tf_val, calc_vals = prepare_input(models[key], A, B, temps, elem_props)
                preds = models[key].predict(X)
                
                # Store debug info for THIS model
                all_debug_data[cfg['name']] = {
                    "vals": calc_vals,
                    "features": X.iloc[0].to_dict()
                }
                
                # --- PLOTLY CONFIG (16:9 RATIO) ---
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=temps, y=preds,
                    mode='lines+markers',
                    line=dict(width=4, color=cfg['color']),
                    marker=dict(size=8, color=cfg['color']),
                ))
                
                # Y-Label: Full Property Name + Unit
                y_label_full = f"<b>{cfg['name']} ({cfg['unit']})</b>"
                
                fig.update_layout(
                    title=dict(
                        text=f"<b>{cfg['name']}</b>",
                        x=0.5,
                        font=dict(size=20, color="black", family="Arial Black")
                    ),
                    xaxis=dict(
                        title=dict(text="<b>Temperature (K)</b>", font=dict(size=18, color="black")),
                        tickfont=dict(size=14, color="black"),
                        showgrid=True, gridcolor='#E0E0E0',
                        showline=True, linewidth=2, linecolor='black',
                        mirror=True, ticks="outside", tickcolor="black", tickwidth=2
                    ),
                    yaxis=dict(
                        title=dict(text=y_label_full, font=dict(size=18, color="black")),
                        tickfont=dict(size=14, color="black"),
                        showgrid=True, gridcolor='#E0E0E0',
                        showline=True, linewidth=2, linecolor='black',
                        mirror=True, ticks="outside", tickcolor="black", tickwidth=2
                    ),
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    margin=dict(l=80, r=20, t=60, b=60),
                    height=360, # FIXED HEIGHT FOR 16:9
                )
                
                with grid_locs[idx]:
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                idx += 1

        status_msg = f"Tolerance Factor: {tf_val:.3f} | Stable Structure"
        
        # --- DEBUG LOGS (FOR ALL MODELS) ---
        with st.expander("Show Debug Logs (Internal Calculations)", expanded=False):
            st.markdown("### Composition Analysis")
            d1, d2 = st.columns(2)
            d1.write("**A-Site:** " + str(A))
            d2.write("**B-Site:** " + str(B))
            
            st.divider()
            
            # Create Tabs for each Model to show their specific features
            if all_debug_data:
                tabs = st.tabs(list(all_debug_data.keys()))
                for i, model_name in enumerate(all_debug_data.keys()):
                    with tabs[i]:
                        data = all_debug_data[model_name]
                        
                        st.markdown(f"#### Calculated Descriptors for {model_name}")
                        st.dataframe(pd.DataFrame.from_dict(data['vals'], orient='index', columns=['Value']))
                        
                        st.markdown(f"#### Feature Vector (First Row) for {model_name}")
                        st.dataframe(pd.DataFrame([data['features']]))

    except Exception as e:
        status_msg = f"Error: {str(e)}"
        st.error(str(e))

# --- STATUS BAR ---
st.markdown(f'<div class="status-bar">{status_msg}</div>', unsafe_allow_html=True)
