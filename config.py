"""
Configuration for P2-ETF-FRACTAL-COMPLEXITY engine.
"""

import os
from datetime import datetime

# --- Hugging Face Repositories ---
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-fractal-complexity-results"

# --- Universe Definitions ---
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]

EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM"
]

ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}

# --- Data Periods ---
DAILY_LOOKBACK = 252                  # Days for daily trading
GLOBAL_TRAIN_START = "2008-01-01"     # Start date for global training

# --- Complexity Parameters ---
ROLLING_WINDOW = 63                   # Days for correlation estimation
LZIV_NORMALIZE = True                 # Normalize Lempel‑Ziv complexity to [0,1]
SAMPLE_ENTROPY_M = 2                  # Embedding dimension
SAMPLE_ENTROPY_R = 0.2                # Tolerance (fraction of std)
TSALLIS_Q = 1.5                       # q parameter (non‑extensive)
MIN_OBSERVATIONS = 252                # Minimum data required (for daily)
GLOBAL_MIN_OBSERVATIONS = 1008        # Minimum data required for global (4 years)

# --- Scoring ---
WEIGHT_LZ = 0.33                      # Weight for Lempel‑Ziv contribution
WEIGHT_SAMPEN = 0.33                  # Weight for Sample Entropy contribution
WEIGHT_TSALLIS = 0.34                 # Weight for Tsallis contribution

# --- Expected Return ---
RETURN_LOOKBACK = 21                  # Days for raw expected return

# --- Shrinking Windows (optional) ---
SHRINKING_WINDOW_START_YEARS = list(range(2010, 2025))

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
