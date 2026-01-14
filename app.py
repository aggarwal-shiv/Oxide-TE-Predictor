import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import os
import plotly.graph_objects as go

# 1. Page config first (must be very early)
st.set_page_config(page_title="Oxide TE-Predictor", layout="wide")

# 2. CSS / styling
st.markdown("""<style>...</style>""", unsafe_allow_html=True)

# 3. ALL FUNCTION DEFINITIONS come BEFORE calling them
@st.cache_data
def load_resources():
    elem_props = {}
    if os.path.exists("data/elemental_properties.xlsx"):
        df = pd.read_excel("data/elemental_properties.xlsx")
        df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        elem_props = df.set_index("Element").T.to_dict()
    return elem_props

@st.cache_resource
def load_models():
    # your model loading code...
    pass

def parse_formula(formula):
    # your parsing code...
    pass

def prepare_input(model, A, B, T, elem_props):
    # your prepare code...
    pass

# 4. NOW it's safe to call them
elem_props = load_resources()   # ← now this will work
models = load_models()

# ... rest of your app (header, input, button logic, etc.)
