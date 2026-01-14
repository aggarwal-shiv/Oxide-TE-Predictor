import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import os
import matplotlib.pyplot as plt

# =============================================================================
# 0. NAMESPACE PATCH (CRITICAL FOR PICKLE LOADING)
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

# =============================================================================
# 1. DEBUG CONFIGURATION
# =============================================================================
DEBUG_FEATURE_LOG = True   

# =============================================================================
# 2. SITE DEFINITIONS
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
# 3. GLOBAL FONT & PLOT STYLE
# =============================================================================
TARGET_FONT = "sans-serif"
plt.rcParams.update({
    "font.family": TARGET_FONT,
    "font.weight": "bold",
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 2.5,
    "xtick.major.width": 2.5,
    "ytick.major.width": 2.5,
    "lines.linewidth": 4.0,
    "lines.markersize": 0
})

# =============================================================================
# 4. CONFIGURATION
# =============================================================================
st.set_page_config(page_title="Perovskite TE Predictor", layout="wide")
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
    "HE": "Heat_of_Evaporation_kJ_per_mol",
    "HF": "Heat_of_Fusion_kJ_per_mol",
}

MODELS_CONFIG = {
    "S": { "file": "Seebeck_Coefficient_S_μV_K__ExtraTrees.pkl", "name": "Seebeck Coefficient", "unit": "μV/K", "color": "#1f77b4" },
    "Sigma": { "file": "Electrical_Conductivity_σ_S_cm__CatBoost.pkl", "name": "Electrical Conductivity", "unit": "S/cm", "color": "#ff7f0e" },
    "Kappa": { "file": "Thermal_Conductivity_κ_W_m-K__GradientBoost.pkl", "name": "Thermal Conductivity", "unit": "W/m·K", "color": "#2ca02c" },
    "zT": { "file": "Figure_of_Merit_zT_CatBoost.pkl", "name": "Figure of Merit (zT)", "unit": "", "color": "#d62728" }
}

# =============================================================================
# 5. CORE LOGIC
# =============================================================================
@st.cache_data
def load_element_database():
    if not os.path.exists(PROPERTIES_DB_PATH):
        st.error(f"Elemental property database not found at {PROPERTIES_DB_PATH}")
        return None
    df = pd.read_excel(PROPERTIES_DB_PATH)
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    return df.set_index("Element").T.to_dict()

def load_models():
    models = {}
    for key, cfg in MODELS_CONFIG.items():
        path = os.path.join(BASE_MODEL_DIR, cfg["file"])
        if not os.path.exists(path):
            st.warning(f"Model file missing: {cfg['file']}")
            continue
        try:
            with open(path, "rb") as f:
                models[key] = pickle.load(f)
        except Exception:
            st.warning(f"Failed to load {cfg['name']}")
    return models

def parse_formula(formula):
    pattern = re.compile(r"([A-Z][a-z]*)(\d*\.?\d*)")
    parts = pattern.findall(formula)
    if not parts: raise ValueError("Invalid chemical formula.")
    
    elements = {}
    for el, amt in parts:
        amt = float(amt) if amt else 1.0
        elements[el] = elements.get(el, 0.0) + amt

    A_site, B_site = {}, {}
    for el, amt in elements.items():
        if el in X_SITE_ELEMENTS: continue
        elif el in A_SITE_ELEMENTS: A_site[el] = amt
        elif el in B_SITE_ELEMENTS: B_site[el] = amt
        else: raise ValueError(f"Element '{el}' is not defined for A-site or B-site.")

    if not A_site: raise ValueError("No valid A-site elements detected.")
    if not B_site: raise ValueError("No valid B-site elements detected.")

    # --- STRICT SUM CHECK (NO NORMALIZATION) ---
    sum_a = sum(A_site.values())
    sum_b = sum(B_site.values())
    
    # We use a small epsilon for float comparison errors (e.g. 0.9999999)
    if abs(sum_a - 1.0) > 0.01:
        raise ValueError(f"A-site stoichiometry must sum to 1.0. Current sum: {sum_a:.2f}")
    
    if abs(sum_b - 1.0) > 0.01:
        raise ValueError(f"B-site stoichiometry must sum to 1.0. Current sum: {sum_b:.2f}")

    return A_site, B_site

def prepare_input(model, A, B, T, elem_props, debug_container=None, model_key=""):
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
    
    X = pd.DataFrame(data)

    if DEBUG_FEATURE_LOG and debug_container:
        with debug_container:
            st.markdown(f"**[DEBUG] MODEL: {model_key}**")
            c1, c2 = st.columns(2)
            with c1:
                st.write("A-site composition:", A)
                st.write("B-site composition:", B)
            with c2:
                st.write("Calculated values (First Row):")
                st.dataframe(pd.DataFrame.from_dict(vals, orient="index", columns=["Value"]).style.format("{:.4f}"))
            st.divider()

    return X, tf

# =============================================================================
# 6. UI APPLICATION
# =============================================================================
st.title("Perovskite TE Predictor")

col1, col2 = st.columns([3, 1])
with col1:
    formula_entry = st.text_input("Formula", value="La0.2Ca0.8TiO3", label_visibility="collapsed")
with col2:
    btn = st.button("Analyze", type="primary", use_container_width=True)

elem_props = load_element_database()
models = load_models()

if btn and elem_props:
    try:
        # 1. Parse (STRICT MODE NOW ACTIVE)
        A, B = parse_formula(formula_entry.strip())
        temps = np.arange(300, 1101, 50)
        
        st.subheader(f"Prediction for: {formula_entry}")
        
        # 2. Setup Plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        debug_box = st.expander("Show Debug Logs", expanded=False) if DEBUG_FEATURE_LOG else None
        
        tf_val = 0
        plot_idx = 0
        plot_order = ["S", "Sigma", "Kappa", "zT"]
        
        # 3. Predict & Plot
        for key in plot_order:
            if key in models:
                cfg = MODELS_CONFIG[key]
                model = models[key]
                
                X, tf_val = prepare_input(model, A, B, temps, elem_props, debug_box, key)
                preds = model.predict(X)
                
                ax = axes[plot_idx]
                ax.plot(temps, preds, color=cfg["color"])
                ax.fill_between(temps, preds.min(), preds, color=cfg["color"], alpha=0.15)
                ax.set_title(cfg["name"])
                ax.set_xlabel("Temperature (K)")
                ylabel = cfg["name"] + (f" ({cfg['unit']})" if cfg["unit"] else "")
                ax.set_ylabel(ylabel)
                ax.grid(True, linestyle="--", alpha=0.6)
                plot_idx += 1
        
        for i in range(plot_idx, 4): axes[i].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        status = "Stable Structure" if 0.75 <= tf_val <= 1.15 else "Unstable / Distorted"
        st.success(f"Tolerance Factor: {tf_val:.3f} | {status} | A-site: {A} | B-site: {B}")

    except ValueError as ve:
        # Specific error for Stoichiometry issues
        st.error(f"Stoichiometry Error: {str(ve)}")
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")

