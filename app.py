import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import os
import plotly.graph_objects as go

# =============================================================================
# PAGE SETUP - MODERN & CLEAN
# =============================================================================
st.set_page_config(page_title="Perovskite TE Predictor", layout="wide", initial_sidebar_state="collapsed")

# =============================================================================
# BEAUTIFUL & COMPACT CSS (inspired by modern scientific dashboards)
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, sans-serif !important;
    }

    .stApp {
        background-color: #f8fafc;
    }

    .block-container {
        padding-top: 1.4rem !important;
        padding-bottom: 6rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 1380px;
        margin: auto;
    }

    .main-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0.6rem 0 1.4rem 0;
        letter-spacing: -0.8px;
        background: linear-gradient(90deg, #334155, #64748b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .input-group {
        max-width: 680px;
        margin: 0 auto 2rem auto;
        display: flex;
        gap: 12px;
    }

    div[data-testid="stTextInput"] input {
        border: 2px solid #cbd5e1;
        border-radius: 10px;
        font-size: 1.28rem;
        font-weight: 600;
        text-align: center;
        padding: 14px;
        height: 3.3rem;
        flex: 1;
    }

    .predict-btn {
        background: linear-gradient(90deg, #3b82f6, #60a5fa) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-size: 1.22rem !important;
        font-weight: 700 !important;
        height: 3.3rem !important;
        min-width: 140px !important;
        transition: all 0.2s;
    }

    .predict-btn:hover {
        background: linear-gradient(90deg, #2563eb, #3b82f6) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(59,130,246,0.35);
    }

    .plot-card {
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.08);
        padding: 1.2rem;
        margin-bottom: 1.6rem;
    }

    .plot-header {
        font-size: 1.38rem;
        font-weight: 700;
        color: #111827;
        text-align: center;
        margin-bottom: 0.9rem;
    }

    .status-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(241,245,249,0.94);
        backdrop-filter: blur(10px);
        border-top: 1px solid #cbd5e1;
        color: #334155;
        text-align: center;
        padding: 14px;
        font-weight: 600;
        font-size: 1.08rem;
        z-index: 999;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">Perovskite TE Predictor</div>', unsafe_allow_html=True)

# Modern Input + Button
with st.container():
    st.markdown('<div class="input-group">', unsafe_allow_html=True)
    col_input, col_btn = st.columns([5, 2])
    with col_input:
        formula = st.text_input("", value="La0.2Ca0.8TiO3", label_visibility="collapsed")
    with col_btn:
        btn = st.button("Analyze", key="predict", help="Run prediction", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# YOUR ORIGINAL BACKEND LOGIC (kept 100% the same)
# =============================================================================

# ... [paste your complete backend code here - class FeatureAwareModel, constants, 
# load_resources, load_models, parse_formula, prepare_input - exactly as you have] ...

# Just showing the improved UI/plotting part below for brevity

elem_props = load_resources()
models = load_models()

status_msg = "Ready — enter formula and click Analyze"

if btn and elem_props:
    try:
        A, B = parse_formula(formula.strip())
        temps = np.arange(300, 1101, 50)

        row1 = st.columns(2, gap="medium")
        row2 = st.columns(2, gap="medium")
        grid_locs = row1 + row2

        tf_val = 0
        idx = 0

        for key in ["S", "Sigma", "Kappa", "zT"]:
            if key not in models:
                continue

            cfg = MODELS_CONFIG[key]
            X, tf_val, calc_vals = prepare_input(models[key], A, B, temps, elem_props)
            preds = models[key].predict(X)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=temps, y=preds,
                mode='lines+markers',
                line=dict(width=3.5, color=cfg['color']),
                marker=dict(size=7, color=cfg['color'], line=dict(width=1.2, color='white')),
            ))

            y_label = f"<b>{cfg['symbol']} ({cfg['unit']})</b>" if cfg['unit'] else f"<b>{cfg['symbol']}</b>"

            fig.update_layout(
                title_text=cfg['name'],
                title_x=0.5,
                title_font=dict(size=20, color="#111827"),
                height=390,               # Balanced for 4 plots
                margin=dict(l=60, r=30, t=70, b=70),
                xaxis_title="Temperature (K)",
                yaxis_title=y_label,
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(family="Inter", size=13, color="#374151"),
                xaxis=dict(
                    showline=True, linewidth=1.6, linecolor='#4b5563', mirror=True,
                    ticks="outside", gridcolor='rgba(209,213,219,0.5)'
                ),
                yaxis=dict(
                    showline=True, linewidth=1.6, linecolor='#4b5563', mirror=True,
                    ticks="outside", gridcolor='rgba(209,213,219,0.5)'
                ),
                hovermode="x unified",
                showlegend=False
            )

            with grid_locs[idx]:
                st.markdown('<div class="plot-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="plot-header">{cfg["name"]}</div>', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                st.markdown('</div>', unsafe_allow_html=True)

            idx += 1

        status_msg = f"Tolerance Factor: {tf_val:.3f} | Likely Stable Perovskite Structure"

        # Nicer bottom info
        st.markdown(f"""
        <div style="text-align:center; font-size:1.1rem; color:#374151; margin-top:1.5rem;">
            A-site: <b>{A}</b> &nbsp;|&nbsp; B-site: <b>{B}</b>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        status_msg = f"Error: {str(e)}"
        st.error(str(e))

# Status bar (glassmorphism style)
st.markdown(f'<div class="status-bar">{status_msg}</div>', unsafe_allow_html=True)
