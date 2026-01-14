import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import os
import plotly.graph_objects as go

# =============================================================================
# PAGE CONFIG - must be first thing using st
# =============================================================================
st.set_page_config(
    page_title="Oxide TE-Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# COMPACT + MODERN CSS - much less wasted space
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, sans-serif !important;
    }

    .stApp {
        background-color: #f9fafb;
    }

    /* Very tight top margin - almost no wasted space above title */
    .block-container {
        padding-top: 1.1rem !important;
        padding-bottom: 5.5rem !important;
        padding-left: 1.6rem !important;
        padding-right: 1.6rem !important;
        max-width: 1380px;
        margin: auto;
    }

    /* Title - big, bold, no extra space */
    .main-header {
        text-align: center;
        font-size: clamp(2.1rem, 5vw, 3.1rem);
        font-weight: 800;
        margin: 0.25rem 0 0.9rem 0 !important;
        padding: 0 !important;
        color: #0f172a;
        letter-spacing: -0.8px;
    }

    /* Input + Button area - bigger & tighter */
    div[data-testid="stTextInput"] > div > div > input {
        border: 2px solid #9ca3af;
        border-radius: 10px;
        font-size: 1.25rem !important;
        font-weight: 600;
        padding: 14px 12px !important;
        height: 3.1rem;
        text-align: center;
        background-color: white;
    }

    div.stButton > button {
        background: linear-gradient(90deg, #2563eb, #3b82f6);
        color: white;
        border: none;
        border-radius: 10px;
        font-size: 1.22rem;
        font-weight: 700;
        height: 3.1rem;
        margin-top: 0.1rem;
        transition: all 0.2s;
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #1d4ed8, #2563eb);
        transform: translateY(-1px);
        box-shadow: 0 5px 15px rgba(37,99,235,0.35);
    }

    /* Plot containers - compact & beautiful */
    .stPlotlyChart {
        background: white;
        border-radius: 10px;
        box-shadow: 0 3px 14px rgba(0,0,0,0.08);
        margin-bottom: 0.9rem !important;
    }

    /* Better mobile stacking */
    @media (max-width: 780px) {
        .row-widget.stHorizontal {
            flex-direction: column !important;
        }
        .stPlotlyChart {
            height: 320px !important;
        }
    }

    .status-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(243,244,246,0.94);
        backdrop-filter: blur(10px);
        border-top: 1px solid #d1d5db;
        color: #374151;
        text-align: center;
        padding: 12px;
        font-weight: 600;
        font-size: 1.02rem;
        z-index: 999;
    }
</style>
""", unsafe_allow_html=True)

# Title - very little space above & below
st.markdown('<div class="main-header">Oxide TE-Predictor</div>', unsafe_allow_html=True)

# Input + Button - tight together
c1, c2, c3 = st.columns([1.5, 3.5, 1.5], gap="small")
with c2:
    formula = st.text_input("", value="La0.2Ca0.8TiO3", label_visibility="collapsed")
with c3:
    btn = st.button("Predict", use_container_width=True)

# ────────────────────────────────────────────────
#  Rest of your original logic (unchanged except plots)
# ────────────────────────────────────────────────

# ... [keep all your original imports, class FeatureAwareModel, constants, functions: 
# load_resources, load_models, parse_formula, prepare_input ]

# Just replace the plotting part with this better version:

if btn and 'elem_props' in locals():
    try:
        A, B = parse_formula(formula.strip())
        temps = np.arange(300, 1101, 50)

        # 2×2 grid but very compact
        row1 = st.columns(2, gap="medium")
        row2 = st.columns(2, gap="medium")
        cols = row1 + row2

        plot_idx = 0

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
                line=dict(width=3.2, color=cfg['color']),
                marker=dict(size=7.5, color=cfg['color'], line=dict(width=1.1,color='white')),
            ))

            fig.update_layout(
                # Modern, clean, bigger fonts
                title=dict(
                    text=cfg['name'],
                    x=0.5,
                    font=dict(size=23, color='#111827', family="Inter"),
                    pad=dict(t=8, b=6)
                ),
                xaxis_title=dict(text="Temperature (K)", font=dict(size=16, color='#374151')),
                yaxis_title=dict(text=f"{cfg['unit']}", font=dict(size=16, color='#374151')),
                
                # Very compact layout - helps fit 4 plots on screen
                height=340,                     # ← smaller but still readable
                margin=dict(l=55, r=25, t=55, b=55),
                
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                
                font=dict(family="Inter", color="#1f2937", size=13),
                showlegend=False,
                hovermode="x unified",
                
                # Proper spines (axes lines)
                xaxis=dict(
                    showline=True, linewidth=1.4, linecolor='#4b5563',
                    mirror=True,
                    ticks="outside", tickwidth=1.3, tickcolor='#4b5563',
                    gridcolor='rgba(209,213,219,0.6)',
                    zeroline=False
                ),
                yaxis=dict(
                    showline=True, linewidth=1.4, linecolor='#4b5563',
                    mirror=True,
                    ticks="outside", tickwidth=1.3, tickcolor='#4b5563',
                    gridcolor='rgba(209,213,219,0.6)',
                    zeroline=False
                )
            )

            with cols[plot_idx]:
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            plot_idx += 1

        status_msg = f"✓ Tolerance factor: {tf_val:.3f}  |  Likely stable perovskite"

    except Exception as e:
        status_msg = f"Error: {str(e)}"
        st.error(str(e))

else:
    status_msg = "Enter formula and click Predict"

st.markdown(f'<div class="status-bar">{status_msg}</div>', unsafe_allow_html=True)
