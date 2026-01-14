import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import os
import plotly.graph_objects as go
import traceback

# ────────────────────────────────────────────────
#               PAGE CONFIGURATION
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Oxide TE-Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ────────────────────────────────────────────────
#             MODERN + COMPACT STYLE
# ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif !important; }
    .block-container { padding-top: 1.1rem !important; padding-bottom: 5.5rem !important;
                       padding-left: 1.6rem !important; padding-right: 1.6rem !important;
                       max-width: 1380px; margin: auto; }
    .main-header { text-align: center; font-size: 2.5rem; font-weight: 800;
                   margin: 0.4rem 0 1.1rem 0; color: #0f172a; letter-spacing: -0.4px; }
    div[data-testid="stTextInput"] input { border: 2px solid #9ca3af; border-radius: 8px;
                                            font-size: 1.22rem; font-weight: 600; text-align: center;
                                            padding: 12px; height: 3.1rem; }
    div.stButton > button { background: linear-gradient(90deg, #2563eb, #3b82f6); color: white;
                            border: none; border-radius: 8px; font-size: 1.2rem; font-weight: 700;
                            height: 3.1rem; margin-top: 0.2rem; }
    div.stButton > button:hover { background: linear-gradient(90deg, #1d4ed8, #2563eb); }
    .stPlotlyChart { background: white; border-radius: 10px;
                     box-shadow: 0 3px 14px rgba(0,0,0,0.08); margin-bottom: 0.9rem !important; }
    .status-bar { position: fixed; bottom: 0; left: 0; right: 0;
                  background: rgba(243,244,246,0.96); backdrop-filter: blur(10px);
                  border-top: 1px solid #d1d5db; color: #374151; text-align: center;
                  padding: 12px; font-weight: 600; font-size: 1.05rem; z-index: 999; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Oxide TE-Predictor</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────
#                   INPUT AREA
# ────────────────────────────────────────────────
c1, c2, c3 = st.columns([1.8, 4.2, 1.8], gap="small")
with c2:
    formula = st.text_input("", value="La0.2Ca0.8TiO3", label_visibility="collapsed")
with c3:
    btn = st.button("Predict", use_container_width=True)

# ────────────────────────────────────────────────
#               DEBUG SECTION (collapsible)
# ────────────────────────────────────────────────
with st.expander("🔍 Debug Information (click to expand)", expanded=False):
    st.write("**Current working directory:**", os.getcwd())
    st.write("**final_models folder exists?**", os.path.exists("final_models"))
    st.write("**elemental_properties.xlsx exists?**", os.path.exists("data/elemental_properties.xlsx"))

    if os.path.exists("final_models"):
        st.write("**Files in final_models:**")
        st.write(os.listdir("final_models"))
    else:
        st.warning("final_models folder NOT found!")

# ────────────────────────────────────────────────
#             MODEL & DATA LOADING LOGIC
# ────────────────────────────────────────────────

class FeatureAwareModel:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = list(feature_names)

    def predict(self, X):
        return self.model.predict(X[self.feature_names])

    def get_feature_names(self):
        return self.feature_names

import __main__
setattr(__main__, "FeatureAwareModel", FeatureAwareModel)

BASE_MODEL_DIR = "final_models"
PROPERTIES_DB_PATH = "data/elemental_properties.xlsx"

PROP_MAP = {
    "Z":   "Atomic_Number",
    "IE":  "Ionization_Energy_kJ_per_mol",
    "EN":  "Electronegativity_Pauling",
    "EA":  "Electron_Affinity_kJ_per_mol",
    "IR":  "Ionic_Radius_pm",
    "MP":  "Melting_Point_C",
    "BP":  "Boiling_Point_C",
    "AD":  "Atomic_Density_g_per_cm3",
    "HE": "Heat_of_Evaporation_kJ_per_mol",   # Most common correct name
    "HF": "Heat_of_Fusion_kJ_per_mol"
}

A_SITE = {"Ca","Sr","Ba","Pb","La","Nd","Sm","Gd","Dy","Ho","Eu","Pr","Na","K","Ce","Bi","Er","Yb","Cu","Y","In","Sb"}
B_SITE = {"Ti","Zr","Nb","Co","Mn","Fe","W","Sn","Hf","Ni","Ta","Ir","Mo","Ru","Rh","Cr"}
X_SITE = {"O"}

MODELS_CONFIG = {
    "S":     {"file": "Seebeck_Coefficient_S_μV_K__ExtraTrees.pkl",     "name": "Seebeck Coefficient",     "unit": "µV/K",      "color": "#1f77b4"},
    "Sigma": {"file": "Electrical_Conductivity_σ_S_cm__CatBoost.pkl",   "name": "Electrical Conductivity", "unit": "S/cm",      "color": "#ff7f0e"},
    "Kappa": {"file": "Thermal_Conductivity_κ_W_m-K__GradientBoost.pkl","name": "Thermal Conductivity",   "unit": "W/m·K",     "color": "#2ca02c"},
    "zT":    {"file": "Figure_of_Merit_zT_CatBoost.pkl",                "name": "Figure of Merit (zT)",    "unit": " ",         "color": "#d62728"}
}

@st.cache_data
def load_resources():
    if not os.path.exists(PROPERTIES_DB_PATH):
        return {}
    try:
        df = pd.read_excel(PROPERTIES_DB_PATH)
        df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        return df.set_index("Element").T.to_dict()
    except Exception as e:
        st.error(f"Error loading elemental properties: {str(e)}")
        return {}

@st.cache_resource
def load_models():
    models = {}
    for k, cfg in MODELS_CONFIG.items():
        path = os.path.join(BASE_MODEL_DIR, cfg["file"])
        if not os.path.exists(path):
            st.warning(f"Model file not found: {path}")
            continue
        try:
            with open(path, "rb") as f:
                models[k] = pickle.load(f)
        except Exception as e:
            st.warning(f"Failed to load model {k}: {str(e)}")
    return models

def parse_formula(formula):
    pattern = re.compile(r"([A-Z][a-z]*)(\d*\.?\d*)")
    parts = pattern.findall(formula)
    if not parts:
        raise ValueError("Invalid formula format")

    elements = {}
    for el, amt in parts:
        amt = float(amt) if amt else 1.0
        elements[el] = elements.get(el, 0) + amt

    A, B = {}, {}
    for el, amt in elements.items():
        if el in X_SITE: continue
        if el in A_SITE: A[el] = amt
        elif el in B_SITE: B[el] = amt
        else:
            raise ValueError(f"Unsupported element: {el}")

    if abs(sum(A.values()) - 1.0) > 0.06:
        raise ValueError(f"A-site sum = {sum(A.values()):.3f} (should ≈ 1.0)")
    if abs(sum(B.values()) - 1.0) > 0.06:
        raise ValueError(f"B-site sum = {sum(B.values()):.3f} (should ≈ 1.0)")

    return A, B

def prepare_input(model, A, B, T, elem_props):
    req = model.get_feature_names()
    N = len(T)
    vals = {}
    for p, col in PROP_MAP.items():
        vals[f"{p}_A"] = sum(elem_props.get(e, {}).get(col, 0) * r for e, r in A.items())
        vals[f"{p}_B"] = sum(elem_props.get(e, {}).get(col, 0) * r for e, r in B.items())

    tf = (vals.get("IR_A", 0) + 140) / (1.414 * (vals.get("IR_B", 0) + 140))
    vals["Tf"] = vals["τ"] = tf

    data = {col: T if col == "T" else np.full(N, vals.get(col, 0)) for col in req}
    return pd.DataFrame(data), tf, vals

# ────────────────────────────────────────────────
#                   MAIN LOGIC
# ────────────────────────────────────────────────
elem_props = load_resources()
models = load_models()

status_msg = "Ready — enter formula and click Predict"

if btn:
    st.write("**Debug:** Button clicked")  # ← basic confirmation

    if not elem_props:
        status_msg = "Missing elemental data"
        st.error("Cannot read data/elemental_properties.xlsx")
    elif not models:
        status_msg = "No models loaded"
        st.error("Check final_models/ folder — missing or corrupted .pkl files?")
    else:
        try:
            st.write("**Debug:** Starting formula parsing...")
            A, B = parse_formula(formula.strip())
            st.write("**Debug:** Formula parsed → A:", A, "B:", B)

            temps = np.arange(300, 1101, 50)

            row1 = st.columns(2, gap="medium")
            row2 = st.columns(2, gap="medium")
            cols = row1 + row2

            for i, key in enumerate(["S", "Sigma", "Kappa", "zT"]):
                if key not in models:
                    st.info(f"Model {key} not available")
                    continue

                st.write(f"**Debug:** Processing {key}...")
                cfg = MODELS_CONFIG[key]
                X, tf_val, _ = prepare_input(models[key], A, B, temps, elem_props)
                preds = models[key].predict(X)

                st.write(f"**Debug:** Predictions ready for {key} (length: {len(preds)})")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=temps, y=preds,
                    mode='lines+markers',
                    line=dict(width=3.2, color=cfg["color"]),
                    marker=dict(size=7, color=cfg["color"], line=dict(width=1.1, color='white')),
                ))

                fig.update_layout(
                    title=dict(text=cfg["name"], x=0.5, font=dict(size=20, color="#111827")),
                    xaxis_title=dict(text="Temperature (K)", font=dict(size=15, color="#374151")),
                    yaxis_title=dict(text=cfg["unit"], font=dict(size=15, color="#374151")),
                    height=340,
                    margin=dict(l=50, r=30, t=55, b=55),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    font=dict(family="Inter", size=12.5, color="#374151"),
                    xaxis=dict(showline=True, linewidth=1.5, linecolor='#4b5563', mirror=True,
                               ticks="outside", tickwidth=1.4, gridcolor='rgba(209,213,219,0.55)'),
                    yaxis=dict(showline=True, linewidth=1.5, linecolor='#4b5563', mirror=True,
                               ticks="outside", tickwidth=1.4, gridcolor='rgba(209,213,219,0.55)'),
                    showlegend=False,
                    hovermode="x unified"
                )

                with cols[i]:
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            status_msg = f"Tolerance factor: {tf_val:.3f} • Likely stable perovskite"

        except Exception as e:
            status_msg = "Error occurred — see details below"
            st.error(f"**Error:** {str(e)}")
            st.code(traceback.format_exc(), language="python")

# ────────────────────────────────────────────────
#                   STATUS BAR
# ────────────────────────────────────────────────
st.markdown(f'<div class="status-bar">{status_msg}</div>', unsafe_allow_html=True)



