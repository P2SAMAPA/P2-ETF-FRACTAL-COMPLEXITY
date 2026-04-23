# P2-ETF-FRACTAL-COMPLEXITY

**Fractal Regime Complexity Engine – Lempel‑Ziv, Sample Entropy, and Tsallis Entropy for Systemic Risk Scoring**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-FRACTAL-COMPLEXITY/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-FRACTAL-COMPLEXITY/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--fractal--complexity--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-fractal-complexity-results)

## Overview

`P2-ETF-FRACTAL-COMPLEXITY` measures systemic complexity using **Lempel‑Ziv complexity**, **Sample Entropy**, and **Tsallis entropy** on rolling correlation matrices. Each ETF's contribution to total complexity is computed, and expected returns are penalized for high systemic importance, ranking ETFs that reduce portfolio fragility.

## Methodology

- **Rolling correlation matrices (63‑day window)** track market structure.
- **Lempel‑Ziv** quantifies pattern diversity in correlations.
- **Sample Entropy** gauges correlation predictability.
- **Tsallis entropy (q=1.5)** captures tail‑correlation complexity.
- **ETF contributions** computed via leave‑one‑out complexity reduction.
- **Adjusted Return** = Raw Return × (1 – 0.5 × Complexity Contribution).

## Universe
FI/Commodities, Equity Sectors, Combined (23 ETFs)

## Dashboard Features

- **Two Main Tabs**: **Daily Trading** (last 252 days) and **Global Training** (full 2008–2026 dataset).
- **Hero Cards**: Top pick with adjusted return and systemic complexity contribution.
- **Top 3 Picks Table**: Detailed breakdown of LZ, Sample Entropy, and Tsallis contributions.
- **All ETFs Table**: Full ranking with raw returns, complexity contributions, and adjusted returns.
- **Explanation Expander**: How complexity metrics are computed and used.
- **Next Trading Day**: U.S. market calendar integration.

