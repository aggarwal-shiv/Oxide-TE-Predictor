import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import os
import sys
import matplotlib.pyplot as plt

# --- 1. CRITICAL IMPORTS FOR MODELS ---
# Even if you don't use them directly, pickle needs these to be loaded
try:
    import sklearn
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
    import catboost 
    import xgboost
except ImportError as e:
    st.error(f"Missing Library: {e}. Please install it using pip.")

# =============================================================================
# 2. CLASS DEFINITION & NAMESPACE PATCH
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

# --- PATCH: Trick Pickle into finding the class in the right place ---
# This maps the class to "__main__" so the pickle file finds it.
import __main__
setattr(__main__, "FeatureAwareModel", FeatureAwareModel)

# =============================================================================
# 3. SITE DEFINITIONS
# =============================================================================
A_SITE_ELEMENTS = {
    "Ca","Sr","Ba","Pb","La","Nd","Sm","Gd","Dy","Ho","Eu","Pr",
    "Na","K","Ce","Bi","Er","Yb","Cu","Y","In","Sb"
}
B_SITE_ELEMENTS = {
    "Ti","Zr","Nb","Co","Mn","Fe","W","Sn","Hf",
    "Ni","Ta","Ir","Mo","Ru","Rh","Cr"
}
X_SITE_ELEMENTS = {"O"}

# =============================================================================
# 4. CONFIGURATION
# =============================================================================
st.set_page_config(page_title="Perovskite TE Predictor", layout="wide")

BASE_MODEL_DIR = "final_models"
PROPERTIES_DB_PATH = "data/elemental_properties.xlsx"

PROP_MAP = {
    "Z": "Atomic_Number", "IE": "Ionization_Energy_kJ_per_mol",
    "EN": "Electronegativity_Pauling", "EA": "Electron_Affinity_kJ_per_mol",
    "IR": "Ionic_Radius_pm", "MP": "Melting_Point_C", "BP": "Boiling_Point_C",
    "AD": "Atomic_Density_g_per_cm3", "HoE": "Heat_of_Evaporation_kJ_per_mol",
    "HoF": "Heat_of_Fusion_kJ_per_mol",
}

MODELS_CONFIG = {
    "S": { "file": "Seebeck_Coefficient_S_μV_K__ExtraTrees.pkl", "name": "Seebeck Coefficient", "unit": "μV/K", "color": "#1f77b4" },
    "Sigma": { "file": "Electrical_Conductivity_σ_S_cm__CatBoost.pkl", "name": "Electrical Conductivity", "unit": "S/cm", "color": "#ff7f0e" },
    "Kappa": { "file": "Thermal_Conductivity_κ_W_m-K__GradientBoost.pkl", "name": "Thermal Conductivity", "unit": "W/m·K", "color": "#2ca02c" },
    "zT": { "file": "Figure_of_Merit_zT_CatBoost.pkl", "name": "Figure of Merit (zT)", "unit": "", "color": "#d62728" }
}

# =============================================================================
# 5. LOADING FUNCTIONS (With Debugging)
# =============================================================================
@st.cache_data
def load_element_database():
    if not os.path.exists(PROPERTIES_DB_PATH):
        st.error(f"DATABASE MISSING: {PROPERTIES_DB_PATH}")
        return None
    df = pd.read_excel(PROPERTIES_DB_PATH)
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    return df.set_index("Element").T.to_dict()

# REMOVED @st.cache_resource to prevent caching errors with custom classes
def load_models():
    models = {}
    for key, cfg in MODELS_CONFIG.items():
        path = os.path.join(BASE_MODEL_DIR, cfg["file"])
        
        if not os.path.exists(path):
            st.error(f"❌ MISSING: {cfg['file']}")
            continue

        try:
            with open(path, "rb") as f:
                models[key] = pickle.load(f)
        except Exception as e:
            # Detailed Error Diagnosis
            f_size = os.path.getsize(path) / (1024 * 1024) # Size in MB
            st.error(f"❌ ERROR loading {cfg['file']} ({f_size:.2f} MB)")
            st.error(f"Details: {e}")
            
    return models

# =============================================================================
# 6. LOGIC & UI (Same as before)
# =============================================================================
def parse_formula(formula):
    pattern = re.compile(r"([A-Z][a-z]*)(\d*\.?\d*)")
    parts = pattern.findall(formula)
    if not parts: raise ValueError("Invalid chemical formula format.")
    
    elements = {}
    for el, amt in parts:
        amt = float(amt) if amt else 1.0
        elements[el] = elements.get(el, 0.0) + amt

    A_site, B_site = {}, {}
    for el, amt in elements.items():
        if el in X_SITE_ELEMENTS: continue
        elif el in A_SITE_ELEMENTS: A_site[el] = amt
        elif el in B_SITE_ELEMENTS: B_site[el] = amt
        else: raise ValueError(f"Element '{el}' not defined for A/B sites.")

    if not A_site or not B_site: raise ValueError("Invalid A or B site elements.")
    
    A_sum = sum(A_site.values())
    B_sum = sum(B_site.values())
    A_norm = {k: v / A_sum for k, v in A_site.items()}
    B_norm = {k: v / B_sum for k, v in B_site.items()}
    return A_norm, B_norm, A_sum, B_sum

def prepare_input(model, A, B, T, elem_props):
    req_features = model.get_feature_names()
    N = len(T)
    vals = {}
    for p_key, col in PROP_MAP.items():
        vals[f"{p_key}_A"] = sum(elem_props[e][col] * r for e, r in A.items())
        vals[f"{p_key}_B"] = sum(elem_props[e][col] * r for e, r in B.items())
        
    tf = (vals["IR_A"] + 140.0) / (1.414 * (vals["IR_B"] + 140.0))
    vals["Tf"], vals["τ"] = tf, tf
    
    data = {}
    for col in req_features:
        if col == "T": data[col] = T
        elif col in vals: data[col] = np.full(N, vals[col])
        else: data[col] = np.zeros(N)
    return pd.DataFrame(data), tf

st.title("Perovskite TE Predictor")

with st.sidebar:
    st.header("Settings")
    formula_input = st.text_input("Chemical Formula", value="La0.2Ca0.8TiO3")
    run_btn = st.button("Analyze Composition", type="primary")

elem_props = load_element_database()
models = load_models()

if run_btn and elem_props:
    try:
        A, B, sum_A, sum_B = parse_formula(formula_input.strip())
        st.subheader(f"Results for: {formula_input}")
        
        temps = np.arange(300, 1101, 50)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        tf_val = 0
        plot_idx = 0
        
        plot_order = ["S", "Sigma", "Kappa", "zT"]
        for key in plot_order:
            if key in models:
                cfg = MODELS_CONFIG[key]
                X, tf_val = prepare_input(models[key], A, B, temps, elem_props)
                preds = models[key].predict(X)
                
                ax = axes[plot_idx]
                ax.plot(temps, preds, color=cfg["color"], linewidth=3)
                ax.fill_between(temps, preds.min(), preds, color=cfg["color"], alpha=0.15)
                ax.set_title(cfg["name"])
                ax.set_ylabel(cfg["unit"])
                ax.grid(True, linestyle="--")
                plot_idx += 1
                
        for i in range(plot_idx, 4): axes[i].axis('off')
        plt.tight_layout()
        st.pyplot(fig)
        
        status = "Stable" if 0.75 <= tf_val <= 1.15 else "Unstable"
        st.success(f"Tolerance Factor: {tf_val:.3f} ({status}) | A-Sum: {sum_A:.2f} | B-Sum: {sum_B:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
