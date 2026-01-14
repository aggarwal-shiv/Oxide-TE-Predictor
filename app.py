import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import os
import matplotlib.pyplot as plt

# =============================================================================
# 1. CLASS DEFINITION (REQUIRED FOR PICKLE LOADING)
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

# =============================================================================
# 2. CONFIGURATION & CONSTANTS
# =============================================================================
st.set_page_config(page_title="Perovskite TE Predictor", layout="wide")

# (Your original element definitions)
A_SITE_ELEMENTS = {
    "Ca","Sr","Ba","Pb","La","Nd","Sm","Gd","Dy","Ho","Eu","Pr",
    "Na","K","Ce","Bi","Er","Yb","Cu","Y","In","Sb"
}
B_SITE_ELEMENTS = {
    "Ti","Zr","Nb","Co","Mn","Fe","W","Sn","Hf",
    "Ni","Ta","Ir","Mo","Ru","Rh","Cr"
}
X_SITE_ELEMENTS = {"O"}

# Directories (Relative paths for web deployment)
BASE_MODEL_DIR = "final_models"
PROPERTIES_DB_PATH = "data/elemental_properties.xlsx"

PROP_MAP = {
    "Z":  "Atomic_Number",
    "IE": "Ionization_Energy_kJ_per_mol",
    "EN": "Electronegativity_Pauling",
    "EA": "Electron_Affinity_kJ_per_mol",
    "IR": "Ionic_Radius_pm",
    "MP": "Melting_Point_C",
    "BP": "Boiling_Point_C",
    "AD": "Atomic_Density_g_per_cm3",
    "HoE": "Heat_of_Evaporation_kJ_per_mol",
    "HoF": "Heat_of_Fusion_kJ_per_mol",
}

MODELS_CONFIG = {
    "S": {"file": "Seebeck_Coefficient_S_μV_K__ExtraTrees.pkl", "name": "Seebeck Coefficient", "unit": "μV/K", "color": "#1f77b4"},
    "Sigma": {"file": "Electrical_Conductivity_σ_S_cm__CatBoost.pkl", "name": "Electrical Conductivity", "unit": "S/cm", "color": "#ff7f0e"},
    "Kappa": {"file": "Thermal_Conductivity_κ_W_m-K__GradientBoost.pkl", "name": "Thermal Conductivity", "unit": "W/m·K", "color": "#2ca02c"},
    "zT": {"file": "Figure_of_Merit_zT_CatBoost.pkl", "name": "Figure of Merit (zT)", "unit": "", "color": "#d62728"}
}

# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================
@st.cache_resource
def load_data_and_models():
    # Load Properties
    if not os.path.exists(PROPERTIES_DB_PATH):
        st.error(f"File not found: {PROPERTIES_DB_PATH}")
        return None, None
    
    df = pd.read_excel(PROPERTIES_DB_PATH)
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    elem_props = df.set_index("Element").T.to_dict()
    
    # Load Models
    models = {}
    for key, cfg in MODELS_CONFIG.items():
        path = os.path.join(BASE_MODEL_DIR, cfg["file"])
        if not os.path.exists(path):
            st.error(f"Model missing: {cfg['file']}")
            continue
        with open(path, "rb") as f:
            models[key] = pickle.load(f)
            
    return elem_props, models

def parse_formula(formula):
    pattern = re.compile(r"([A-Z][a-z]*)(\d*\.?\d*)")
    parts = pattern.findall(formula)
    if not parts:
        raise ValueError("Invalid chemical formula format.")
        
    elements = {}
    for el, amt in parts:
        amt = float(amt) if amt else 1.0
        elements[el] = elements.get(el, 0.0) + amt
        
    A_site, B_site = {}, {}
    for el, amt in elements.items():
        if el in X_SITE_ELEMENTS: continue
        elif el in A_SITE_ELEMENTS: A_site[el] = amt
        elif el in B_SITE_ELEMENTS: B_site[el] = amt
        else: raise ValueError(f"Element '{el}' not recognized for A/B sites.")
            
    if not A_site or not B_site:
        raise ValueError("Must have both A-site and B-site elements.")
        
    # Normalize
    A_site = {k: v / sum(A_site.values()) for k, v in A_site.items()}
    B_site = {k: v / sum(B_site.values()) for k, v in B_site.items()}
    return A_site, B_site

def prepare_input(model, A, B, T, elem_props):
    req_features = model.get_feature_names()
    N = len(T)
    vals = {}
    for p_key, col in PROP_MAP.items():
        vals[f"{p_key}_A"] = sum(elem_props[e][col] * r for e, r in A.items())
        vals[f"{p_key}_B"] = sum(elem_props[e][col] * r for e, r in B.items())
        
    rO = 140.0
    tf = (vals["IR_A"] + rO) / (1.414 * (vals["IR_B"] + rO))
    vals["Tf"] = tf
    vals["τ"] = tf
    
    data = {}
    for col in req_features:
        if col == "T": data[col] = T
        elif col in vals: data[col] = np.full(N, vals[col])
        else: data[col] = np.zeros(N)
            
    return pd.DataFrame(data), tf

# =============================================================================
# 4. MAIN APP INTERFACE
# =============================================================================
st.title("Perovskite Thermoelectric Predictor")
st.markdown("Enter a perovskite composition to predict TE properties vs Temperature.")

# Sidebar for inputs
with st.sidebar:
    st.header("Configuration")
    formula = st.text_input("Chemical Formula", value="La0.2Ca0.8TiO3")
    run_btn = st.button("Analyze Composition", type="primary")

# Load resources
elem_props, models = load_data_and_models()

if run_btn and elem_props and models:
    try:
        # 1. Parse
        A, B = parse_formula(formula)
        temps = np.arange(300, 1101, 50)
        
        # 2. Setup Plot
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes_flat = axes.flatten()
        
        # 3. Predict & Plot
        tf_val = 0
        plot_idx = 0
        
        # Mapping config keys to plot positions
        key_order = ["S", "Sigma", "Kappa", "zT"]
        
        for key in key_order:
            if key not in models: continue
            
            cfg = MODELS_CONFIG[key]
            model = models[key]
            
            X, tf_val = prepare_input(model, A, B, temps, elem_props)
            preds = model.predict(X)
            
            ax = axes_flat[plot_idx]
            ax.plot(temps, preds, color=cfg["color"], linewidth=3)
            ax.fill_between(temps, preds.min(), preds, color=cfg["color"], alpha=0.15)
            ax.set_title(cfg["name"], fontsize=12, fontweight='bold')
            ax.set_ylabel(f"{cfg['name']} [{cfg['unit']}]" if cfg['unit'] else cfg['name'])
            ax.grid(True, linestyle="--", alpha=0.6)
            plot_idx += 1

        # 4. Final Layout Adjustments
        for ax in axes_flat:
            if not ax.has_data(): 
                ax.axis('off') # Hide empty plots if any
            else:
                ax.set_xlabel("Temperature (K)")

        plt.tight_layout()
        st.pyplot(fig)
        
        # 5. Status Output
        status = "Stable Structure" if 0.75 <= tf_val <= 1.15 else "Unstable / Distorted"
        st.info(f"**Tolerance Factor:** {tf_val:.3f} | **Status:** {status}")
        with st.expander("View Composition Details"):
            st.write("A-site:", A)
            st.write("B-site:", B)

    except Exception as e:
        st.error(f"Error: {e}")