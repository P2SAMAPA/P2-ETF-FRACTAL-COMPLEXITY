"""
Main training script for Fractal Regime Complexity Engine.
Computes both daily (252d) and global (2008‑present) metrics.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from fractal_complexity_model import FractalComplexityModel
import push_results

def compute_universe_results(returns: pd.DataFrame, model: FractalComplexityModel, 
                             expected_returns: pd.Series = None) -> tuple:
    """
    Compute contributions, adjusted returns, and top picks for a given returns DataFrame.
    Returns (universe_results, top_picks, metrics_df)
    """
    metrics_df = model.compute_complexity_metrics(returns)
    contributions = model.compute_etf_contributions(returns)
    
    if expected_returns is None:
        expected_returns = model.compute_expected_return(returns)
    adj_returns = model.compute_complexity_adjusted_return(expected_returns, contributions)

    universe_results = {}
    for ticker in returns.columns:
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

    sorted_tickers = sorted(universe_results.items(),
                            key=lambda x: x[1]["expected_return_adj"], reverse=True)
    top_picks = [
        {k: v for k, v in d.items() if k != 'ticker'} | {"ticker": t}
        for t, d in sorted_tickers[:3]
    ]
    return universe_results, top_picks, metrics_df


def run_fractal_complexity():
    print(f"=== P2-ETF-FRACTAL-COMPLEXITY Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()
    df_master = df_master[df_master['Date'] >= config.GLOBAL_TRAIN_START]

    model = FractalComplexityModel(
        window=config.ROLLING_WINDOW,
        lziv_normalize=config.LZIV_NORMALIZE,
        sample_m=config.SAMPLE_ENTROPY_M,
        sample_r=config.SAMPLE_ENTROPY_R,
        tsallis_q=config.TSALLIS_Q
    )

    daily_results = {"universes": {}, "top_picks": {}, "complexity_history": {}}
    global_results = {"universes": {}, "top_picks": {}, "complexity_history": {}}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        returns = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns) < config.MIN_OBSERVATIONS:
            continue

        # ---- Daily (recent 252 days) ----
        daily_returns = returns.iloc[-config.DAILY_LOOKBACK:]
        if len(daily_returns) >= config.MIN_OBSERVATIONS:
            print("  Computing daily metrics...")
            daily_exp = model.compute_expected_return(daily_returns)
            daily_univ, daily_top, daily_hist = compute_universe_results(daily_returns, model, daily_exp)
            daily_results["universes"][universe_name] = daily_univ
            daily_results["top_picks"][universe_name] = daily_top
            daily_results["complexity_history"][universe_name] = daily_hist.to_dict(orient='list')
        else:
            print(f"  Skipping daily (only {len(daily_returns)} observations)")

        # ---- Global (full history) ----
        if len(returns) >= config.GLOBAL_MIN_OBSERVATIONS:
            print("  Computing global metrics...")
            global_exp = model.compute_expected_return(returns)
            global_univ, global_top, global_hist = compute_universe_results(returns, model, global_exp)
            global_results["universes"][universe_name] = global_univ
            global_results["top_picks"][universe_name] = global_top
            global_results["complexity_history"][universe_name] = global_hist.to_dict(orient='list')
        else:
            print(f"  Skipping global (only {len(returns)} observations)")

    output_payload = {
        "run_date": config.TODAY,
        "config": {k: v for k, v in config.__dict__.items() if not k.startswith("_") and k.isupper() and k != "HF_TOKEN"},
        "daily": daily_results,
        "global": global_results
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")


if __name__ == "__main__":
    run_fractal_complexity()
