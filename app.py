import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import os
import plotly.graph_objects as go

# =============================================================================
# 1. PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Oxide TE-Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# 2. GLOBAL STYLES (POLISHED UI)
# =============================================================================
st.markdown("""
<style>
/* ---------------- GLOBAL ---------------- */
.stApp {
    background-color: #F4F7F9;
    font-family: Arial, Helvetica, sans-serif;
    color: #000000 !important;
}

.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1400px;
    margin: auto;
}

div[data-testid="stVerticalBlock"] > div {
    gap: 0.6rem !important;
}

/* ---------------- HEADER ---------------- */
.custom-header {
    text-align: center;
    font-size: 34px;
    font-weight: 800;
    color: #0F172A;
    margin-top: 8px;
    margin-bottom: 2px;
}

.sub-header {
    text-align: center;
    font-size: 15px;
    color: #475569;
    margin-bottom: 10px;
}

/* ---------------- INPUT BAR ---------------- */
div[data-testid="stTextInput"] input {
    border: 2px solid #334155;
    border-right: none;
    border-radius: 4px 0 0 4px;
    text-align: center;
    font-weight: 700;
    font-size: 17px;
    height: 42px;
    background: #FFFFFF;
}

div.stButton > button {
    background-color: #0052CC;
    color: white;
    border: 2px solid #0052CC;
    border-radius: 0 4px 4px 0;
    font-weight: 700;
    height: 42px;
    font-size: 15px;
}

div.stButton > button:hover {
    background-color: #003E99;
    border-color: #003E99;
}

/* ---------------- PLOT CARDS ---------------- */
.plot-card {
    background: white;
    border-radius: 6px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.08);
    padding: 8px;
}

/* ---------------- STATUS BAR ---------------- */
.status-bar {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #F1F5F9;
    border-top: 2px solid #334155;
    padding: 8px 30px;
    font-size: 14px;
    font-weight: 700;
    color: #334155;
    z-index: 9999;
}

/* ---------------- HIDE STREAMLIT ---------------- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 3. BACKEND LOGIC (UNCHANGED)
# =============================================================================
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
    "S": {"file": "Seebeck_Coefficient_S_μV_K__ExtraTrees.pkl", "name": "Seebeck Coefficient", "symbol": "S", "unit": "µV·K⁻¹", "color": "#1f77b4"},
    "Sigma": {"file": "Electrical_Conductivity_σ_S_cm__CatBoost.pkl", "name": "Electrical Conductivity", "symbol": "σ", "unit": "S·cm⁻¹", "color": "#ff7f0e"},
    "Kappa": {"file": "Thermal_Conductivity_κ_W_m-K__GradientBoost.pkl", "name": "Thermal Conductivity", "symbol": "κ", "unit": "W·m⁻¹·K⁻¹", "color": "#2ca02c"},
    "zT": {"file": "Figure_of_Merit_zT_CatBoost.pkl", "name": "Figure of Merit", "symbol": "zT", "unit": "", "color": "#d62728"}
}

@st.cache_data
def load_element_props():
    df = pd.read_excel(PROPERTIES_DB_PATH)
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    return df.set_index("Element").T.to_dict()

@st.cache_resource
def load_models():
    models = {}
    for k, cfg in MODELS_CONFIG.items():
        with open(os.path.join(BASE_MODEL_DIR, cfg["file"]), "rb") as f:
            models[k] = pickle.load(f)
    return models

def parse_formula(formula):
    pattern = re.compile(r"([A-Z][a-z]*)(\d*\.?\d*)")
    parts = pattern.findall(formula)
    elements = {}
    for el, amt in parts:
        elements[el] = elements.get(el, 0) + (float(amt) if amt else 1.0)
    A, B = {}, {}
    for el, amt in elements.items():
        if el in X_SITE: continue
        elif el in A_SITE: A[el] = amt
        elif el in B_SITE: B[el] = amt
        else: raise ValueError(f"Unknown element: {el}")
    A = {k: v/sum(A.values()) for k, v in A.items()}
    B = {k: v/sum(B.values()) for k, v in B.items()}
    return A, B

def prepare_input(model, A, B, T, props):
    req = model.get_feature_names()
    vals = {}
    for p, col in PROP_MAP.items():
        vals[f"{p}_A"] = sum(props[e][col]*r for e, r in A.items())
        vals[f"{p}_B"] = sum(props[e][col]*r for e, r in B.items())
    tf = (vals["IR_A"] + 140) / (1.414*(vals["IR_B"] + 140))
    vals["Tf"], vals["τ"] = tf, tf
    data = {c: (T if c=="T" else np.full(len(T), vals.get(c, 0))) for c in req}
    return pd.DataFrame(data), tf

# =============================================================================
# 4. UI
# =============================================================================
st.markdown('<div class="custom-header">Oxide TE-Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Machine-Learning-Driven Thermoelectric Property Prediction of Oxide Perovskites</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns([3, 4, 2])
with c2:
    formula = st.text_input("", "La0.2Ca0.8TiO3")
with c3:
    run = st.button("Predict")

props = load_element_props()
models = load_models()
status_msg = "System Ready"

if run:
    A, B = parse_formula(formula)
    T = np.arange(300, 1101, 50)
    row1 = st.columns(2)
    row2 = st.columns(2)
    grid = row1 + row2

    tf_val = 0
    for i, key in enumerate(["S","Sigma","Kappa","zT"]):
        X, tf_val = prepare_input(models[key], A, B, T, props)
        preds = models[key].predict(X)

        cfg = MODELS_CONFIG[key]
        fig = go.Figure(go.Scatter(
            x=T, y=preds,
            mode="lines+markers",
            line=dict(width=3, color=cfg["color"]),
            marker=dict(size=6)
        ))

        fig.update_layout(
            title=dict(text=f"<b>{cfg['name']}</b>", x=0.5, y=0.88),
            height=400,
            margin=dict(l=60, r=20, t=50, b=40),
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=dict(title="<b>Temperature (K)</b>", showline=True, mirror=True),
            yaxis=dict(title=f"<b>{cfg['symbol']} ({cfg['unit']})</b>" if cfg["unit"] else "<b>zT</b>", showline=True, mirror=True)
        )

        with grid[i]:
            st.markdown('<div class="plot-card">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

    status_msg = f"Tolerance Factor: {tf_val:.3f} | A-site: {A} | B-site: {B}"

st.markdown(f'<div class="status-bar">{status_msg}</div>', unsafe_allow_html=True)
