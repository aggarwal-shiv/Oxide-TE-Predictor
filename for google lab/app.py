from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import re
import os
from feature_model import FeatureAwareModel


# ============================================================
# DEBUG CONFIGURATION
# ============================================================
DEBUG_FEATURE_LOG = False   # Must be False for web



# ============================================================
# SITE DEFINITIONS (UNCHANGED)
# ============================================================
A_SITE_ELEMENTS = {
    "Ca","Sr","Ba","Pb","La","Nd","Sm","Gd","Dy","Ho","Eu","Pr",
    "Na","K","Ce","Bi","Er","Yb","Cu","Y","In","Sb"
}

B_SITE_ELEMENTS = {
    "Ti","Zr","Nb","Co","Mn","Fe","W","Sn","Hf",
    "Ni","Ta","Ir","Mo","Ru","Rh","Cr"
}

X_SITE_ELEMENTS = {"O"}

# ============================================================
# CONFIGURATION
# ============================================================
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
    "S": "Seebeck_Coefficient_S_μV_K__ExtraTrees.pkl",
    "Sigma": "Electrical_Conductivity_σ_S_cm__CatBoost.pkl",
    "Kappa": "Thermal_Conductivity_κ_W_m-K__GradientBoost.pkl",
    "zT": "Figure_of_Merit_zT_CatBoost.pkl",
}

# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__)

# ============================================================
# LOAD ELEMENT DATABASE
# ============================================================
df = pd.read_excel(PROPERTIES_DB_PATH)
df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
ELEM_PROPS = df.set_index("Element").T.to_dict()

# ============================================================
# LOAD MODELS (ONCE)
# ============================================================
MODELS = {}
for key, fname in MODELS_CONFIG.items():
    with open(os.path.join(BASE_MODEL_DIR, fname), "rb") as f:
        MODELS[key] = pickle.load(f)

# ============================================================
# FORMULA PARSER (STRICT & IDENTICAL)
# ============================================================
def parse_formula(formula):
    pattern = re.compile(r"([A-Z][a-z]*)(\d*\.?\d*)")
    parts = pattern.findall(formula)
    if not parts:
        raise ValueError("Invalid chemical formula.")

    elements = {}
    for el, amt in parts:
        amt = float(amt) if amt else 1.0
        elements[el] = elements.get(el, 0.0) + amt

    A_site, B_site = {}, {}

    for el, amt in elements.items():
        if el in X_SITE_ELEMENTS:
            continue
        elif el in A_SITE_ELEMENTS:
            A_site[el] = amt
        elif el in B_SITE_ELEMENTS:
            B_site[el] = amt
        else:
            raise ValueError(f"Element '{el}' not defined for A/B site.")

    if not A_site or not B_site:
        raise ValueError("Invalid perovskite composition.")

    A_site = {k: v / sum(A_site.values()) for k, v in A_site.items()}
    B_site = {k: v / sum(B_site.values()) for k, v in B_site.items()}

    return A_site, B_site

# ============================================================
# FEATURE CONSTRUCTION (IDENTICAL)
# ============================================================
def prepare_input(model, A, B, T):
    req = model.get_feature_names()
    N = len(T)
    vals = {}

    for p_key, col in PROP_MAP.items():
        vals[f"{p_key}_A"] = sum(ELEM_PROPS[e][col] * r for e, r in A.items())
        vals[f"{p_key}_B"] = sum(ELEM_PROPS[e][col] * r for e, r in B.items())

    rO = 140.0
    tf = (vals["IR_A"] + rO) / (1.414 * (vals["IR_B"] + rO))
    vals["Tf"] = tf
    vals["τ"] = tf

    data = {}
    for col in req:
        if col == "T":
            data[col] = T
        elif col in vals:
            data[col] = np.full(N, vals[col])
        else:
            data[col] = np.zeros(N)

    return pd.DataFrame(data), tf

# ============================================================
# ROUTES
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    formula = request.json["formula"]
    A, B = parse_formula(formula)
    T = np.arange(300, 1101, 50)

    predictions = {}
    tf_val = None

    for key, model in MODELS.items():
        X, tf_val = prepare_input(model, A, B, T)
        predictions[key] = model.predict(X).tolist()

    return jsonify({
        "temperature": T.tolist(),
        "predictions": predictions,
        "tolerance_factor": tf_val,
        "A_site": A,
        "B_site": B
    })

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    app.run()

