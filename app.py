# =============================================================================
# 1. PAGE SETUP & BETTER MODERN CSS
# =============================================================================
st.set_page_config(
    page_title="Oxide TE-Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"  # better for mobile
)

# Modern, clean, high-contrast style with better typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', system-ui, sans-serif !important;
    }

    .stApp {
        background-color: #f8f9fd;
    }

    /* Main container better padding for mobile & desktop */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 6rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 1400px;
        margin: auto;
    }

    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #1a1a2e;
        letter-spacing: -0.5px;
    }

    /* Header - modern & bold */
    .main-header {
        text-align: center;
        font-size: clamp(1.8rem, 5vw, 3.2rem);
        font-weight: 800;
        margin: 0.4em 0 0.8em 0;
        color: #0f172a;
        background: linear-gradient(90deg, #334155, #64748b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Input + Button area */
    div[data-testid="stTextInput"] > div > div > input {
        border: 2px solid #cbd5e1;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 12px !important;
        background-color: white;
        color: #1e293b;
    }

    div.stButton > button {
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 0.75rem 1.5rem;
        height: 3.2rem;
        transition: all 0.2s;
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #2563eb, #3b82f6);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59,130,246,0.35);
    }

    /* Plot container - better spacing */
    .stPlotlyChart {
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        overflow: hidden;
        margin-bottom: 1.4rem !important;
    }

    /* Responsive grid - better mobile experience */
    @media (max-width: 768px) {
        .row-widget.stHorizontal {
            flex-direction: column !important;
        }
        .stPlotlyChart {
            height: 320px !important;
        }
    }

    /* Footer/status bar - modern glass effect */
    .status-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(241, 245, 249, 0.92);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-top: 1px solid rgba(226,232,240,0.8);
        color: #334155;
        text-align: center;
        padding: 14px;
        font-weight: 600;
        font-size: 1rem;
        z-index: 999;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)

# Better header
st.markdown('<div class="main-header">Oxide TE-Predictor</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────
#                Better Plot Layout Strategy
# ────────────────────────────────────────────────

# Instead of fixed 2×2 grid → more flexible approach

if btn and elem_props:
    try:
        A, B = parse_formula(formula.strip())
        temps = np.arange(300, 1101, 50)

        # We create **two rows** but allow natural flow on mobile
        col1, col2 = st.columns(2, gap="medium")

        plots_placed = 0

        for key in ["S", "Sigma", "Kappa", "zT"]:
            if key not in models:
                continue

            cfg = MODELS_CONFIG[key]
            X, tf_val, calc_vals = prepare_input(models[key], A, B, temps, elem_props)
            preds = models[key].predict(X)

            # ── Modern Plot Style ───────────────────────────────
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=temps, y=preds,
                mode='lines+markers',
                line=dict(width=3.2, color=cfg['color']),
                marker=dict(size=7, color=cfg['color'], line=dict(width=1,color='white')),
                name=cfg['name']
            ))

            fig.update_layout(
                title=dict(
                    text=cfg['name'],
                    x=0.5,
                    font=dict(size=22, color='#1e293b', family="Inter"),
                    pad=dict(t=10,b=10)
                ),
                xaxis_title=dict(
                    text="Temperature (K)",
                    font=dict(size=16, color='#475569')
                ),
                yaxis_title=dict(
                    text=f"{cfg['name']}  ({cfg['unit']})",
                    font=dict(size=16, color='#475569')
                ),
                height=380,               # ← good compromise
                margin=dict(l=60, r=30, t=70, b=60),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#334155"),
                hovermode="x unified",
                showlegend=False,
                xaxis=dict(
                    gridcolor='rgba(226,232,240,0.6)',
                    zeroline=False,
                    linewidth=1.2,
                    linecolor='#cbd5e1'
                ),
                yaxis=dict(
                    gridcolor='rgba(226,232,240,0.6)',
                    zeroline=False,
                    linewidth=1.2,
                    linecolor='#cbd5e1'
                )
            )

            # Put plots in columns (2×2 → responsive)
            target_col = col1 if plots_placed % 2 == 0 else col2

            with target_col:
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={'displayModeBar': False, 'responsive': True}
                )

            plots_placed += 1

        # Final status with tolerance factor
        status_msg = f"✓ Tolerance factor: {tf_val:.3f} | Predicted as stable perovskite"

    except Exception as e:
        status_msg = f"Error: {str(e)}"
        st.error(str(e), icon="🚨")

# Modern status bar
st.markdown(f'<div class="status-bar">{status_msg}</div>', unsafe_allow_html=True)
