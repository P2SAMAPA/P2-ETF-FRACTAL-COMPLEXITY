"""
Fractal Regime Complexity Model using Lempel‑Ziv, Sample Entropy, and Tsallis entropy.
"""

import numpy as np
import pandas as pd
from scipy import stats
import antropy as ant
import config

class FractalComplexityModel:
    def __init__(self, window=63, lziv_normalize=True, sample_m=2, sample_r=0.2, tsallis_q=1.5):
        self.window = window
        self.lziv_normalize = lziv_normalize
        self.sample_m = sample_m
        self.sample_r = sample_r
        self.tsallis_q = tsallis_q

    def compute_correlation_surface(self, returns: pd.DataFrame) -> np.ndarray:
        """Rolling correlation matrices."""
        corrs = []
        for i in range(self.window, len(returns) + 1):
            window_returns = returns.iloc[i-self.window:i]
            corr = window_returns.corr().values
            corrs.append(corr)
        return np.stack(corrs)

    def _flatten_corr(self, corr: np.ndarray) -> np.ndarray:
        """Extract upper triangular elements of correlation matrix."""
        n = corr.shape[0]
        return corr[np.triu_indices(n, k=1)]

    def _lempel_ziv_complexity(self, seq: np.ndarray) -> float:
        """Lempel‑Ziv complexity of a sequence."""
        # Convert to binary string via median threshold
        median = np.median(seq)
        binary = ''.join(['1' if x > median else '0' for x in seq])
        return ant.lziv_complexity(binary, normalize=self.lziv_normalize)

    def _sample_entropy(self, seq: np.ndarray) -> float:
        """Sample entropy of a sequence."""
        if len(seq) < 20:
            return 0.0
        return ant.sample_entropy(seq, order=self.sample_m, metric='chebyshev')

    def _tsallis_entropy(self, seq: np.ndarray) -> float:
        """Tsallis entropy (non‑extensive)."""
        # Estimate probability distribution via histogram
        hist, _ = np.histogram(seq, bins='auto', density=True)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0
        q = self.tsallis_q
        if q == 1:
            return -np.sum(hist * np.log(hist))
        else:
            return (1 - np.sum(hist ** q)) / (q - 1)

    def compute_complexity_metrics(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute complexity metrics over rolling correlation matrices.
        Returns a DataFrame with columns: date, lziv, samp_entropy, tsallis.
        """
        corrs = self.compute_correlation_surface(returns)
        dates = returns.index[self.window-1:]
        metrics = []
        for i, corr in enumerate(corrs):
            flat = self._flatten_corr(corr)
            lziv = self._lempel_ziv_complexity(flat)
            samp = self._sample_entropy(flat)
            tsallis = self._tsallis_entropy(flat)
            metrics.append({
                'date': dates[i],
                'lziv': lziv,
                'samp_entropy': samp,
                'tsallis': tsallis
            })
        return pd.DataFrame(metrics).set_index('date')

    def compute_etf_contributions(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute each ETF's contribution to total complexity reduction.
        Contribution = complexity of full correlation matrix - complexity of matrix without that ETF.
        """
        tickers = returns.columns.tolist()
        n_assets = len(tickers)
        if n_assets < 3:
            return pd.DataFrame()
        
        # Full correlation matrix complexity for the most recent window
        recent_returns = returns.iloc[-self.window:]
        full_corr = recent_returns.corr().values
        full_flat = self._flatten_corr(full_corr)
        
        full_lziv = self._lempel_ziv_complexity(full_flat)
        full_samp = self._sample_entropy(full_flat)
        full_tsallis = self._tsallis_entropy(full_flat)
        
        contributions = []
        for ticker in tickers:
            reduced_returns = recent_returns.drop(columns=[ticker])
            reduced_corr = reduced_returns.corr().values
            reduced_flat = self._flatten_corr(reduced_corr)
            
            reduced_lziv = self._lempel_ziv_complexity(reduced_flat)
            reduced_samp = self._sample_entropy(reduced_flat)
            reduced_tsallis = self._tsallis_entropy(reduced_flat)
            
            # Contribution = reduction in complexity (higher = more systemic importance)
            contrib_lz = full_lziv - reduced_lziv
            contrib_samp = full_samp - reduced_samp
            contrib_tsallis = full_tsallis - reduced_tsallis
            
            contributions.append({
                'ticker': ticker,
                'contrib_lz': contrib_lz,
                'contrib_samp': contrib_samp,
                'contrib_tsallis': contrib_tsallis,
                'composite': (config.WEIGHT_LZ * contrib_lz + 
                              config.WEIGHT_SAMPEN * contrib_samp + 
                              config.WEIGHT_TSALLIS * contrib_tsallis)
            })
        return pd.DataFrame(contributions)

    def compute_expected_return(self, returns: pd.DataFrame) -> pd.Series:
        """Simple expected return: recent 21-day annualized."""
        exp_ret = {}
        for ticker in returns.columns:
            ret = returns[ticker]
            if len(ret) >= 21:
                exp_ret[ticker] = ret.iloc[-21:].mean() * 252
            else:
                exp_ret[ticker] = 0.0
        return pd.Series(exp_ret)

    def compute_complexity_adjusted_return(self, expected_return: pd.Series,
                                           contributions: pd.DataFrame) -> pd.Series:
        """
        Adjust expected return by complexity contribution.
        Higher complexity contribution (systemic importance) receives a negative adjustment,
        as systemic assets are more fragile during regime shifts.
        """
        contrib_map = contributions.set_index('ticker')['composite'].to_dict()
        adj_return = {}
        for ticker, exp in expected_return.items():
            contrib = contrib_map.get(ticker, 0.0)
            # Cap contribution at reasonable values
            contrib = np.clip(contrib, -0.5, 0.5)
            adj_return[ticker] = exp * (1 - 0.5 * contrib)
        return pd.Series(adj_return)
