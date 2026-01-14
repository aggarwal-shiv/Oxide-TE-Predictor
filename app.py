import streamlit as st
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import re
import os
import sys

# =============================================================================
# 1. SETUP & CONFIGURATION
# =============================================================================
app = Flask(__name__)

# --- NAMESPACE PATCH FOR PICKLE ---
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

# --- CONSTANTS ---
BASE_MODEL_DIR = "final_models"
PROPERTIES_DB_PATH = "data/elemental_properties.xlsx"
MODELS_CONFIG = {
    "S": "Seebeck_Coefficient_S_μV_K__ExtraTrees.pkl",
    "Sigma": "Electrical_Conductivity_σ_S_cm__CatBoost.pkl",
    "Kappa": "Thermal_Conductivity_κ_W_m-K__GradientBoost.pkl",
    "zT": "Figure_of_Merit_zT_CatBoost.pkl"
}
PROP_MAP = {"Z":"Atomic_Number", 
            "IE":"Ionization_Energy_kJ_per_mol", 
            "EN":"Electronegativity_Pauling", 
            "EA":"Electron_Affinity_kJ_per_mol",
            "IR":"Ionic_Radius_pm", 
            "MP":"Melting_Point_C", 
            "BP":"Boiling_Point_C", 
            "AD":"Atomic_Density_g_per_cm3", 
            "HE":"Heat_of_Evaporation_kJ_per_mol", 
            "HF":"Heat_of_Fusion_kJ_per_mol"}
A_SITE = {"Ca","Sr","Ba","Pb","La","Nd","Sm","Gd","Dy","Ho","Eu","Pr","Na","K","Ce","Bi","Er","Yb","Cu","Y","In","Sb"}
B_SITE = {"Ti","Zr","Nb","Co","Mn","Fe","W","Sn","Hf","Ni","Ta","Ir","Mo","Ru","Rh","Cr"}
X_SITE = {"O"}

# --- LOAD RESOURCES ---
elem_props = {}
models = {}

def load_resources():
    global elem_props, models
    if os.path.exists(PROPERTIES_DB_PATH):
        df = pd.read_excel(PROPERTIES_DB_PATH)
        df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        elem_props = df.set_index("Element").T.to_dict()
    
    for k, f in MODELS_CONFIG.items():
        path = os.path.join(BASE_MODEL_DIR, f)
        if os.path.exists(path):
            with open(path, "rb") as file:
                models[k] = pickle.load(file)

load_resources()

# --- LOGIC ---
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

    # Strict Check
    if abs(sum(A.values()) - 1.0) > 0.05: raise ValueError(f"A-site sum is {sum(A.values()):.2f}, must be 1.0")
    if abs(sum(B.values()) - 1.0) > 0.05: raise ValueError(f"B-site sum is {sum(B.values()):.2f}, must be 1.0")
    return A, B

def prepare_input(model, A, B, T):
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

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        formula = data.get('formula', '').strip()
        A, B = parse_formula(formula)
        temps = np.arange(300, 1101, 50)
        
        # Prepare Response Structure
        response = {
            "temperature": temps.tolist(),
            "predictions": {},
            "tolerance_factor": 0,
            "A_site": A,
            "B_site": B
        }

        tf_val = 0
        for key in ["S", "Sigma", "Kappa", "zT"]:
            if key in models:
                X, tf_val = prepare_input(models[key], A, B, temps)
                pred = models[key].predict(X)
                response["predictions"][key] = pred.tolist()
        
        response["tolerance_factor"] = tf_val
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # Use 0.0.0.0 for Render/Cloud deployment
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
