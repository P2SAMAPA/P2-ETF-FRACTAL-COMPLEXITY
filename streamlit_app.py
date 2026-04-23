"""
Streamlit Dashboard for Fractal Regime Complexity Engine.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from huggingface_hub import HfApi, hf_hub_download
import json
import config
from us_calendar import USMarketCalendar

st.set_page_config(page_title="P2Quant Fractal Complexity", page_icon="🌀", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 600; color: #1f77b4; }
    .hero-card { background: linear-gradient(135deg, #1f77b4 0%, #2C5282 100%); border-radius: 16px; padding: 2rem; color: white; text-align: center; }
    .hero-ticker { font-size: 4rem; font-weight: 800; }
    .contrib-high { color: #dc3545; font-weight: 600; }
    .contrib-mid { color: #ffc107; font-weight: 600; }
    .contrib-low { color: #28a745; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_latest_results():
    try:
        api = HfApi(token=config.HF_TOKEN)
        files = api.list_repo_files(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset")
        json_files = sorted([f for f in files if f.startswith("fractal_complexity_") and f.endswith('.json')], reverse=True)
        if not json_files:
            return None
        local_path = hf_hub_download(
            repo_id=config.HF_OUTPUT_REPO, filename=json_files[0],
            repo_type="dataset", token=config.HF_TOKEN, cache_dir="./hf_cache"
        )
        with open(local_path) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

def contrib_badge(val):
    if val > 0.1:
        return f'<span class="contrib-high">High Systemic ({val:.3f})</span>'
    elif val > 0.0:
        return f'<span class="contrib-mid">Moderate ({val:.3f})</span>'
    else:
        return f'<span class="contrib-low">Low ({val:.3f})</span>'

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
calendar = USMarketCalendar()
st.sidebar.markdown(f"**📅 Next Trading Day:** {calendar.next_trading_day().strftime('%Y-%m-%d')}")
data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")

st.markdown('<div class="main-header">🌀 P2Quant Fractal Regime Complexity</div>', unsafe_allow_html=True)
st.markdown('<div>Lempel‑Ziv · Sample Entropy · Tsallis Entropy – Systemic Complexity Scoring</div>', unsafe_allow_html=True)

with st.expander("📘 How It Works", expanded=False):
    st.markdown("""
    ### Complexity Metrics
    - **Lempel‑Ziv Complexity**: Measures the number of distinct patterns in correlation sequences.
    - **Sample Entropy**: Quantifies the regularity/predictability of correlation changes.
    - **Tsallis Entropy (q=1.5)**: Non‑extensive entropy sensitive to tail dependencies.
    
    ### ETF Scoring
    Each ETF's **systemic complexity contribution** is computed by removing it from the correlation matrix and measuring the drop in total complexity. Higher contribution = greater systemic importance.
    
    **Adjusted Return** = Raw Return × (1 – 0.5 × Complexity Contribution)
    *Systemic assets receive a penalty due to higher regime‑shift fragility.*
    """)

if data is None:
    st.warning("No data available.")
    st.stop()

daily = data['daily_trading']
universes = daily['universes']
top_picks = daily['top_picks']

tabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
universe_keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]

for tab, key in zip(tabs, universe_keys):
    with tab:
        top = top_picks.get(key, [])
        universe_data = universes.get(key, {})
        if top:
            pick = top[0]
            ticker = pick['ticker']
            ret = pick['expected_return_adj']
            contrib = pick['complexity_contrib']
            st.markdown(f"""
            <div class="hero-card">
                <div style="font-size: 1.2rem; opacity: 0.8;">🌀 TOP PICK (Low Systemic Complexity)</div>
                <div class="hero-ticker">{ticker}</div>
                <div style="font-size: 1.5rem;">Adj Return: {ret*100:.2f}%</div>
                <div style="margin-top: 1rem;">Complexity Contrib: {contrib_badge(contrib)}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Top 3 Picks")
            rows = []
            for p in top:
                rows.append({
                    "Ticker": p['ticker'],
                    "Adj Return": f"{p['expected_return_adj']*100:.2f}%",
                    "Complexity Contrib": f"{p['complexity_contrib']:.4f}",
                    "LZ": f"{p['contrib_lz']:.4f}",
                    "SampEn": f"{p['contrib_samp']:.4f}",
                    "Tsallis": f"{p['contrib_tsallis']:.4f}"
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("### All ETFs")
            all_rows = []
            for t, d in universe_data.items():
                all_rows.append({
                    "Ticker": t,
                    "Raw Return": f"{d['expected_return_raw']*100:.2f}%",
                    "Complexity Contrib": f"{d['complexity_contrib']:.4f}",
                    "Adj Return": f"{d['expected_return_adj']*100:.2f}%"
                })
            df_all = pd.DataFrame(all_rows).sort_values("Adj Return", ascending=False)
            st.dataframe(df_all, use_container_width=True, hide_index=True)
        else:
            st.info(f"No data for {key}")
