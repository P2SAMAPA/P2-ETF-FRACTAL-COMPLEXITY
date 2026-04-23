"""
Main training script for Fractal Regime Complexity Engine.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from fractal_complexity_model import FractalComplexityModel
import push_results

def run_fractal_complexity():
    print(f"=== P2-ETF-FRACTAL-COMPLEXITY Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()

    model = FractalComplexityModel(
        window=config.ROLLING_WINDOW,
        lziv_normalize=config.LZIV_NORMALIZE,
        sample_m=config.SAMPLE_ENTROPY_M,
        sample_r=config.SAMPLE_ENTROPY_R,
        tsallis_q=config.TSALLIS_Q
    )

    all_results = {}
    top_picks = {}
    complexity_history = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        returns = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns) < config.MIN_OBSERVATIONS:
            continue

        recent_returns = returns.iloc[-config.MIN_OBSERVATIONS:]
        
        # Compute complexity metrics history
        metrics_df = model.compute_complexity_metrics(recent_returns)
        complexity_history[universe_name] = metrics_df.to_dict(orient='list')
        
        # Compute ETF contributions
        contributions = model.compute_etf_contributions(recent_returns)
        expected_returns = model.compute_expected_return(recent_returns)
        adj_returns = model.compute_complexity_adjusted_return(expected_returns, contributions)

        universe_results = {}
        for ticker in tickers:
            contrib_row = contributions[contributions['ticker'] == ticker]
            if len(contrib_row) > 0:
                row = contrib_row.iloc[0]
                universe_results[ticker] = {
                    "ticker": ticker,
                    "expected_return_raw": expected_returns.get(ticker, 0.0),
                    "complexity_contrib": row['composite'],
                    "contrib_lz": row['contrib_lz'],
                    "contrib_samp": row['contrib_samp'],
                    "contrib_tsallis": row['contrib_tsallis'],
                    "expected_return_adj": adj_returns.get(ticker, 0.0)
                }
            else:
                universe_results[ticker] = {
                    "ticker": ticker,
                    "expected_return_raw": expected_returns.get(ticker, 0.0),
                    "complexity_contrib": 0.0,
                    "contrib_lz": 0.0,
                    "contrib_samp": 0.0,
                    "contrib_tsallis": 0.0,
                    "expected_return_adj": expected_returns.get(ticker, 0.0)
                }

        all_results[universe_name] = universe_results
        sorted_tickers = sorted(universe_results.items(),
                                key=lambda x: x[1]["expected_return_adj"], reverse=True)
        top_picks[universe_name] = [
            {k: v for k, v in d.items() if k != 'ticker'} | {"ticker": t}
            for t, d in sorted_tickers[:3]
        ]

    output_payload = {
        "run_date": config.TODAY,
        "config": {k: v for k, v in config.__dict__.items() if not k.startswith("_") and k.isupper() and k != "HF_TOKEN"},
        "daily_trading": {
            "universes": all_results,
            "top_picks": top_picks,
            "complexity_history": complexity_history
        }
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_fractal_complexity()
