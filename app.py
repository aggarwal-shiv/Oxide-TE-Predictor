import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import os
import plotly.graph_objects as go

# ────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────
st.set_page_config(page_title="Perovskite TE Predictor", layout="wide", initial_sidebar_state="collapsed")

# ────────────────────────────────────────────────
# MODERN & CLEAN STYLE (closer to your reference image)
# ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, sans-serif !important;
    }

    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 5rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 1400px;
        margin: auto;
    }

    .main-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 800;
        color: #1a1a2e;
        margin: 0.6rem 0 1.2rem 0;
        letter-spacing: -0.5px;
    }

    .input-container {
        max-width: 580px;
        margin: 0 auto 1.8rem auto;
    }

    .input-container input {
        border: 2px solid #cbd5e1;
        border-radius: 10px;
        font-size: 1.25rem;
        font-weight: 600;
        text-align: center;
        padding: 14px;
        height: 3.2rem;
    }

    .analyze-btn {
        background: #3b82f6 !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        height: 3.2rem !important;
        margin-left: 0.8rem !important;
        transition: all 0.2s;
    }

    .analyze-btn:hover {
        background: #2563eb !important;
        transform: translateY(-1px);
    }

    .plot-container {
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        margin-bottom: 1.4rem !important;
        overflow: hidden;
    }

    .plot-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.6rem;
        text-align: center;
    }

    .status-info {
        font-size: 1.05rem;
        color: #374151;
        text-align: center;
        padding: 12px 0;
        border-top: 1px solid #e5e7eb;
        background: #f9fafb;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# HEADER & INPUT
# ────────────────────────────────────────────────
st.markdown('<div class="main-title">Perovskite TE Predictor</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([3, 4, 2])
    with col2:
        formula = st.text_input("", value="La0.2Ca0.8TiO3", label_visibility="collapsed")
    with col3:
        btn = st.button("Analyze", key="analyze", help="Click to predict", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────
# (Your original model loading, parsing, prepare_input functions here)
# ... paste your complete backend code here (class, constants, load_resources, load_models, parse_formula, prepare_input) ...
# For brevity, I'm showing only the UI/plotting part below
# ────────────────────────────────────────────────

# Example placeholder – replace with your actual logic
if btn:
    try:
        # Your actual parsing & prediction code here
        A, B = parse_formula(formula.strip())
        temps = np.arange(300, 1101, 50)

        # Dummy predictions for demonstration (replace with real ones)
        preds_S = -60 - 0.1 * (temps - 300)
        preds_Sigma = 2500 * np.exp(-0.005 * (temps - 300))
        preds_Kappa = 4.1 - 0.0012 * (temps - 300)
        preds_zT = 0.07 + 0.00032 * (temps - 300)

        tf_val = 0.793

        # ── 2×2 PLOTS ────────────────────────────────────────────────
        row1 = st.columns(2, gap="medium")
        row2 = st.columns(2, gap="medium")
        all_cols = row1 + row2

        plots_data = [
            ("Seebeck Coefficient", "µV/K", preds_S, "#1f77b4", "S (µV/K)"),
            ("Electrical Conductivity", "S/cm", preds_Sigma, "#ff7f0e", "σ (S/cm)"),
            ("Thermal Conductivity", "W/m·K", preds_Kappa, "#2ca02c", "κ (W/m·K)"),
            ("Figure of Merit (zT)", "", preds_zT, "#d62728", "zT")
        ]

        for i, (name, unit, y_data, color, y_label) in enumerate(plots_data):
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=temps, y=y_data,
                mode='lines+markers',
                line=dict(width=3, color=color),
                marker=dict(size=7, color=color, line=dict(width=1, color='white'))
            ))

            fig.update_layout(
                title_text=name,
                title_x=0.5,
                title_font=dict(size=20, color="#111827"),
                height=380,
                margin=dict(l=60, r=30, t=70, b=70),
                xaxis_title="Temperature (K)",
                yaxis_title=y_label,
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(family="Inter", size=13, color="#374151"),
                xaxis=dict(showline=True, linewidth=1.6, linecolor='#4b5563', mirror=True,
                           ticks="outside", gridcolor='rgba(209,213,219,0.5)'),
                yaxis=dict(showline=True, linewidth=1.6, linecolor='#4b5563', mirror=True,
                           ticks="outside", gridcolor='rgba(209,213,219,0.5)'),
                hovermode="x unified",
                showlegend=False
            )

            with all_cols[i]:
                st.markdown(f'<div class="plot-container">', unsafe_allow_html=True)
                st.markdown(f'<div class="plot-title">{name}</div>', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                st.markdown('</div>', unsafe_allow_html=True)

        # ── BOTTOM INFO ──────────────────────────────────────────────
        st.markdown(f"""
        <div class="status-info">
            Tolerance Factor: <b>{tf_val:.3f}</b> | 
            A-site: <b>{A}</b> | 
            B-site: <b>{B}</b>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error during calculation: {str(e)}")
