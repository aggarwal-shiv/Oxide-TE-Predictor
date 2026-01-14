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
# 3. CONFIGURATION & STYLE
# =============================================================================
st.set_page_config(page_title="Perovskite TE Predictor", layout="wide")

# Custom CSS to mimic your Tkinter fonts
st.markdown("""
<style>
    .big-font { font-size:20px !important; font-weight: bold; }
    .status-box { padding: 15px; border-radius: 5px; background-color: #f0f2f6; border: 1px solid #d1d5db; }
</style>
""", unsafe_allow_html=True)

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
# 4. HELPER FUNCTIONS
# =============================================================================
@st.cache_data
def load_element_database():
    if not os.path.exists(PROPERTIES_DB_PATH):
        st.error(f"DATABASE MISSING: Could not find {PROPERTIES_DB_PATH}")
        return None
    df = pd.read_excel(PROPERTIES_DB_PATH)
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    return df.set_index("Element").T.to_dict()

@st.cache_resource
def load_models():
    models = {}
    errors = []
    for key, cfg in MODELS_CONFIG.items():
        path = os.path.join(BASE_MODEL_DIR, cfg["file"])
        if not os.path.exists(path):
            errors.append(f"Model missing: {cfg['file']}")
            continue
        try:
            with open(path, "rb") as f:
                models[key] = pickle.load(f)
        except Exception as e:
            errors.append(f"Failed to load {cfg['file']}: {str(e)}")
    
    if errors:
        for err in errors: st.error(err)
        
    return models

def parse_formula(formula):
    # Regex to find Element+Number pairs
    pattern = re.compile(r"([A-Z][a-z]*)(\d*\.?\d*)")
    parts = pattern.findall(formula)

    if not parts:
        raise ValueError("Invalid chemical formula format.")

    elements = {}
    for el, amt in parts:
        amt = float(amt) if amt else 1.0
        elements[el] = elements.get(el, 0.0) + amt

    A_site = {}
    B_site = {}
    errors = []

    for el, amt in elements.items():
        if el in X_SITE_ELEMENTS:
            continue
        elif el in A_SITE_ELEMENTS:
            A_site[el] = amt
        elif el in B_SITE_ELEMENTS:
            B_site[el] = amt
        else:
            errors.append(f"Element '{el}' is not defined for A-site or B-site.")

    if errors:
        raise ValueError("\n".join(errors))
    if not A_site:
        raise ValueError("No valid A-site elements detected (e.g., La, Ca, Sr).")
    if not B_site:
        raise ValueError("No valid B-site elements detected (e.g., Ti, Mn, Fe).")

    # Capture raw sums for debugging
    raw_sum_A = sum(A_site.values())
    raw_sum_B = sum(B_site.values())

    # Normalize independently
    A_site_norm = {k: v / raw_sum_A for k, v in A_site.items()}
    B_site_norm = {k: v / raw_sum_B for k, v in B_site.items()}

    return A_site_norm, B_site_norm, raw_sum_A, raw_sum_B

def prepare_input(model, A, B, T, elem_props, debug_log_container=None, model_key=""):
    req_features = model.get_feature_names()
    N = len(T)
    vals = {}

    # Calculate weighted properties
    for p_key, col in PROP_MAP.items():
        try:
            vals[f"{p_key}_A"] = sum(elem_props[e][col] * r for e, r in A.items())
            vals[f"{p_key}_B"] = sum(elem_props[e][col] * r for e, r in B.items())
        except KeyError as e:
            raise ValueError(f"Property data missing for element: {e}")

    # Tolerance Factor
    rO = 140.0
    tf = (vals["IR_A"] + rO) / (1.414 * (vals["IR_B"] + rO))
    vals["Tf"] = tf
    vals["τ"] = tf

    # Build Feature Matrix
    data = {}
    missing_feats = []
    
    for col in req_features:
        if col == "T":
            data[col] = T
        elif col in vals:
            data[col] = np.full(N, vals[col])
        else:
            data[col] = np.zeros(N)
            missing_feats.append(col)

    X = pd.DataFrame(data)

    # --- DEBUG LOGGING (Replicating your _debug_print_features) ---
    if debug_log_container:
        with debug_log_container:
            st.markdown(f"#### 🔍 Debug: {MODELS_CONFIG[model_key]['name']}")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Calculated Properties (A & B Site):**")
                st.dataframe(pd.DataFrame.from_dict(vals, orient="index", columns=["Value"]).style.format("{:.4f}"))
            with c2:
                st.markdown("**First Row of Feature Matrix:**")
                st.dataframe(X.iloc[[0]].T.style.format("{:.4f}"))
            
            if missing_feats:
                st.warning(f"⚠️ Features filled with 0.0 (Not found in calc): {missing_feats}")
            st.divider()

    return X, tf

# =============================================================================
# 5. MAIN UI
# =============================================================================
st.title("Perovskite TE Predictor")

# Sidebar for controls
with st.sidebar:
    st.header("Settings")
    formula_input = st.text_input("Chemical Formula", value="La0.2Ca0.8TiO3")
    show_debug = st.checkbox("Show Detailed Debug Logs", value=False)
    run_btn = st.button("Analyze Composition", type="primary")

# Load Resources
elem_props = load_element_database()
models = load_models()

if run_btn:
    if not elem_props:
        st.stop()

    try:
        # 1. Parse Formula
        A_site, B_site, raw_sum_A, raw_sum_B = parse_formula(formula_input.strip())
        
        # Display Composition Details
        st.subheader(f"Prediction for: {formula_input}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**A-Site Composition** (Sum={raw_sum_A:.2f})\n\n" + str(A_site))
        with col2:
            st.info(f"**B-Site Composition** (Sum={raw_sum_B:.2f})\n\n" + str(B_site))
            
        # 2. Setup Plot
        temps = np.arange(300, 1101, 50)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # Debug Container
        debug_container = st.expander("🛠️ Internal Calculation Logs", expanded=True) if show_debug else None
        
        tf_val = 0
        plot_idx = 0
        
        # 3. Run Models
        # Fixed order to match your 2x2 grid layout preference
        plot_order = ["S", "Sigma", "Kappa", "zT"]
        
        for key in plot_order:
            if key not in models:
                continue
                
            cfg = MODELS_CONFIG[key]
            model = models[key]
            
            # Prepare Input & Log Debug info
            X, tf_val = prepare_input(
                model, A_site, B_site, temps, elem_props, 
                debug_log_container=debug_container, model_key=key
            )
            
            # Predict
            preds = model.predict(X)
            
            # Plot
            ax = axes[plot_idx]
            ax.plot(temps, preds, color=cfg["color"], linewidth=3)
            ax.fill_between(temps, preds.min(), preds, color=cfg["color"], alpha=0.15)
            ax.set_title(cfg["name"], fontsize=14, fontweight='bold')
            ax.set_ylabel(f"{cfg['name']} [{cfg['unit']}]" if cfg['unit'] else cfg['name'])
            ax.grid(True, linestyle="--", alpha=0.6)
            plot_idx += 1

        # Clean up empty plots
        for i in range(plot_idx, 4):
            axes[i].axis('off')
            
        for ax in axes[:plot_idx]:
            ax.set_xlabel("Temperature (K)")

        plt.tight_layout()
        st.pyplot(fig)
        
        # 4. Status Bar (Tolerance Factor)
        with col3:
            status_text = "Stable Structure" if 0.75 <= tf_val <= 1.15 else "Unstable / Distorted"
            status_color = "green" if 0.75 <= tf_val <= 1.15 else "red"
            st.markdown(f"""
            <div class="status-box" style="border-left: 5px solid {status_color}">
                <strong>Tolerance Factor:</strong> {tf_val:.4f}<br>
                <strong>Status:</strong> {status_text}
            </div>
            """, unsafe_allow_html=True)

    except ValueError as ve:
        st.error(f"**Input Error:** {ve}")
    except Exception as e:
        st.error(f"**System Error:** {e}")

