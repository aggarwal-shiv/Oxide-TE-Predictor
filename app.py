import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import os
import plotly.graph_objects as go

# =============================================================================
# 1. VISUAL CONFIGURATION (CSS)
# =============================================================================
st.set_page_config(page_title="Perovskite TE Predictor", layout="wide")

st.markdown("""
<style>
    /* --- GLOBAL SETTINGS --- */
    .stApp {
        background-color: #F0F2F6; 
        font-family: 'Arial', sans-serif;
        color: #000000;
    }

    /* --- LAYOUT CONSTRAINT --- */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 6rem;
        max-width: 1400px;    
        margin: 0 auto;
    }

    /* --- HEADER --- */
    .custom-header {
        text-align: center;
        font-size: 32px;
        font-weight: 800;
        color: #172B4D;
        text-transform: uppercase; 
        margin-bottom: 25px;
        letter-spacing: 0.5px;
        font-family: 'Arial Black', sans-serif;
    }

    /* --- INPUT SECTION --- */
    div[data-testid="stTextInput"] input {
        border: 1px solid #B3B3B3;
        border-radius: 4px 0 0 4px; 
        text-align: center;
        font-weight: bold;
        color: #172B4D;
        font-size: 18px;
        height: 46px;
        background: #FFFFFF;
    }
    
    div.stButton > button {
        background-color: #0052CC; 
        color: white;
        border: none;
        border-radius: 0 4px 4px 0; 
        font-weight: bold;
        height: 46px;
        width: 100%;
        font-size: 16px;
        text-transform: uppercase;
    }
    div.stButton > button:hover {
        background-color: #0747a6;
    }

    /* --- STATUS BAR --- */
    .status-bar {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #E1E4E8;
        border-top: 3px solid #172B4D;
        color: #172B4D;
        text-align: left;
        padding: 12px 30px;
        font-size: 15px;
        font-weight: bold;
        z-index: 9999;
        font-family: 'Arial', sans-serif;
    }

    /* Hide standard elements */
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
    "S": {"file": "Seebeck_Coefficient_S_μV_K__ExtraTrees.pkl", "name": "Seebeck Coefficient", "symbol": "S", "unit": "µV·K⁻¹", "color": "#1f77b4"},
    "Sigma": {"file": "Electrical_Conductivity_σ_S_cm__CatBoost.pkl", "name": "Electrical Conductivity", "symbol": "σ", "unit": "S·cm⁻¹", "color": "#ff7f0e"},
    "Kappa": {"file": "Thermal_Conductivity_κ_W_m-K__GradientBoost.pkl", "name": "Thermal Conductivity", "symbol": "κ", "unit": "W·m⁻¹·K⁻¹", "color": "#2ca02c"},
    "zT": {"file": "Figure_of_Merit_zT_CatBoost.pkl", "name": "Figure of Merit", "symbol": "zT", "unit": "", "color": "#d62728"}
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
    return pd.DataFrame(data), tf, vals

# =============================================================================
# 3. UI LAYOUT
# =============================================================================

# --- HEADER ---
st.markdown('<div class="custom-header">PEROVSKITE TE PREDICTOR</div>', unsafe_allow_html=True)

# --- INPUT BAR ---
c1, c2, c3, c4 = st.columns([3, 4, 1, 3], gap="small")
with c2:
    formula = st.text_input("Formula", value="La0.2Ca0.8TiO3", label_visibility="collapsed")
with c3:
    btn = st.button("Analyze")

# --- MAIN GRID ---
elem_props = load_resources()
models = load_models()
status_msg = "Waiting for input..."

if btn and elem_props:
    try:
        A, B = parse_formula(formula.strip())
        temps = np.arange(300, 1101, 50)
        
        # Larger gap for distinct "Card" separation
        row1 = st.columns(2, gap="large")
        row2 = st.columns(2, gap="large")
        grid_locs = row1 + row2 
        
        tf_val = 0
        idx = 0
        all_debug_data = {}
        
        for key in ["S", "Sigma", "Kappa", "zT"]:
            if key in models:
                cfg = MODELS_CONFIG[key]
                X, tf_val, calc_vals = prepare_input(models[key], A, B, temps, elem_props)
                preds = models[key].predict(X)
                
                all_debug_data[cfg['name']] = {"vals": calc_vals, "features": X.iloc[0].to_dict()}
                
                # Y-Label
                y_label = f"<b>{cfg['symbol']}"
                if cfg['unit']:
                    y_label += f" ({cfg['unit']})"
                y_label += "</b>"
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=temps, y=preds,
                    mode='lines+markers',
                    line=dict(width=3, color=cfg['color']),
                    marker=dict(size=6, color=cfg['color']),
                ))
                
                # --- PRECISE AXIS & FONT CONFIGURATION ---
                fig.update_layout(
                    # Global Font Settings (Arial, Bold, Black)
                    font=dict(family="Arial", size=14, color="black"),
                    
                    # Title
                    title=dict(
                        text=f"<b>{cfg['name']}</b>",
                        x=0.5,
                        y=0.9, # Slight push down
                        font=dict(size=16)
                    ),
                    
                    # X-Axis (Full Box, No Grid, Outward Ticks)
                    xaxis=dict(
                        title=dict(text="<b>Temperature (K)</b>", font=dict(size=14)),
                        showgrid=False,       # NO GRID
                        showline=True,        # Show Axis Line
                        linewidth=2,          # Thicker Line (Spine)
                        linecolor='black',    # Black Spine
                        mirror=True,          # Mirror to Top/Right (Complete Box)
                        ticks="outside",      # Ticks point out
                        tickwidth=2,
                        tickcolor='black',
                        ticklen=6,
                        tickfont=dict(size=12, color="black", family="Arial")
                    ),
                    
                    # Y-Axis (Full Box, No Grid, Outward Ticks)
                    yaxis=dict(
                        title=dict(text=y_label, font=dict(size=14)),
                        showgrid=False,       # NO GRID
                        showline=True,        # Show Axis Line
                        linewidth=2,          # Thicker Line
                        linecolor='black',    # Black Spine
                        mirror=True,          # Mirror to Right (Complete Box)
                        ticks="outside",      # Ticks point out
                        tickwidth=2,
                        tickcolor='black',
                        ticklen=6,
                        tickfont=dict(size=12, color="black", family="Arial")
                    ),
                    
                    # Plot Background
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    
                    # Margins (Prevent label cutting)
                    margin=dict(l=70, r=20, t=50, b=50),
                    
                    # Size (16:9 Aspect Ratio)
                    height=250, 
                )
                
                with grid_locs[idx]:
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                idx += 1

        # Status
        a_str = str(A).replace("'", "").replace("{", "").replace("}", "")
        b_str = str(B).replace("'", "").replace("{", "").replace("}", "")
        status_msg = f"Tolerance Factor: {tf_val:.3f} | A-site: {{{a_str}}} | B-site: {{{b_str}}}"
        
        # --- DEBUG LOGS ---
        with st.expander("Show Debug Logs", expanded=False):
            if all_debug_data:
                tabs = st.tabs(list(all_debug_data.keys()))
                for i, model_name in enumerate(all_debug_data.keys()):
                    with tabs[i]:
                        data = all_debug_data[model_name]
                        c1, c2 = st.columns(2)
                        with c1:
                            st.write("Calculated Properties")
                            st.dataframe(pd.DataFrame.from_dict(data['vals'], orient='index', columns=['Value']))
                        with c2:
                            st.write("Feature Vector")
                            st.dataframe(pd.DataFrame([data['features']]))

    except Exception as e:
        status_msg = f"Error: {str(e)}"
        st.error(str(e))

# --- STATUS BAR ---
st.markdown(f'<div class="status-bar">{status_msg}</div>', unsafe_allow_html=True)
