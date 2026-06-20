"""
Fractal Regime Complexity Model v2.0
- Eigenvalue-based complexity (preserves continuous information)
- Spectral entropy (mathematically correct for matrices)
- Eigenvector centrality contributions (structural importance)
- Adaptive windowing
- Backward compatible with trainer.py
"""

import numpy as np
import pandas as pd
from scipy import linalg
import config


class FractalComplexityModel:
    def __init__(self, window=63, lziv_normalize=None, sample_m=None, sample_r=None, tsallis_q=None,
                 min_window=42, max_window=126, stability_lookback=252):
        # Ignore legacy params (keep for backward compat with trainer.py)
        self.window = window
        self.base_window = window
        self.min_window = min_window
        self.max_window = max_window
        self.stability_lookback = stability_lookback

    def _adaptive_window(self, returns: pd.DataFrame, idx: int) -> int:
        """Adjust window based on correlation stability."""
        start = max(0, idx - self.stability_lookback)
        hist_returns = returns.iloc[start:idx]
        
        if len(hist_returns) < self.base_window * 2:
            return self.base_window
        
        n_check = min(20, len(hist_returns) - self.base_window)
        if n_check < 5:
            return self.base_window
        
        corr_stability = []
        for i in range(n_check):
            end_idx = len(hist_returns) - i
            start_idx = end_idx - self.base_window
            c = hist_returns.iloc[start_idx:end_idx].corr().values
            off_diag = c[np.triu_indices(len(c), k=1)]
            corr_stability.append(np.std(off_diag))
        
        stability = np.std(corr_stability)
        
        if stability > 0.08:
            window = int(self.base_window * 0.7)
        elif stability < 0.03:
            window = int(self.base_window * 1.4)
        else:
            window = self.base_window
        
        return np.clip(window, self.min_window, 
                       min(self.max_window, len(returns.iloc[:idx+1])))
    
    def _eigenvalue_complexity(self, corr: np.ndarray) -> dict:
        """Eigenvalue-based complexity metrics."""
        n = corr.shape[0]
        corr = self._nearest_valid_corr(corr)
        
        eigenvalues = linalg.eigvalsh(corr)
        eigenvalues = np.sort(eigenvalues)[::-1]
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        
        eig_sum = np.sum(eigenvalues)
        p = eigenvalues / eig_sum
        
        # Spectral Entropy
        spectral_entropy = -np.sum(p * np.log(p))
        max_entropy = np.log(n)
        norm_spectral_entropy = spectral_entropy / max_entropy if max_entropy > 0 else 0
        
        # Participation Ratio
        participation_ratio = 1.0 / np.sum(p ** 2)
        max_pr = n
        norm_participation = participation_ratio / max_pr
        
        # Effective Rank
        effective_rank = np.exp(spectral_entropy)
        norm_effective_rank = effective_rank / n
        
        # Dominant Eigenvalue Concentration
        concentration = p[0]
        
        # Eigenvalue Spread
        log_eigenvalues = np.log(eigenvalues)
        eigenvalue_spread = np.std(log_eigenvalues) / np.mean(log_eigenvalues)
        
        return {
            'spectral_entropy': norm_spectral_entropy,
            'participation_ratio': norm_participation,
            'effective_rank': norm_effective_rank,
            'concentration': concentration,
            'eigenvalue_spread': eigenvalue_spread,
            'eigenvalues': eigenvalues
        }
    
    def _nearest_valid_corr(self, corr: np.ndarray, max_iter=100, tol=1e-6) -> np.ndarray:
        """Project to nearest valid correlation matrix."""
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -1, 1)
        
        try:
            np.linalg.cholesky(corr)
            return corr
        except np.linalg.LinAlgError:
            for _ in range(max_iter):
                try:
                    eigenvalues, eigenvectors = np.linalg.eigh(corr)
                    eigenvalues = np.clip(eigenvalues, 1e-8, None)
                    corr = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
                    corr = (corr + corr.T) / 2
                    np.fill_diagonal(corr, 1.0)
                    d = np.sqrt(np.diag(corr))
                    corr = corr / np.outer(d, d)
                    np.fill_diagonal(corr, 1.0)
                    np.linalg.cholesky(corr)
                    return corr
                except np.linalg.LinAlgError:
                    continue
            return corr
    
    def _eigenvector_centrality_contribution(self, corr: np.ndarray) -> np.ndarray:
        """Measure structural importance using eigenvector centrality."""
        n = corr.shape[0]
        corr = self._nearest_valid_corr(corr)
        
        abs_corr = np.abs(corr)
        np.fill_diagonal(abs_corr, 0)
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(abs_corr)
            idx = np.argmax(eigenvalues)
            centrality = np.abs(eigenvectors[:, idx])
            centrality = centrality / np.sum(centrality)
        except:
            centrality = np.ones(n) / n
        
        return centrality
    
    def _marginal_complexity_contribution(self, corr: np.ndarray, asset_idx: int) -> float:
        """How much does removing this asset change the effective rank?"""
        n = corr.shape[0]
        full_metrics = self._eigenvalue_complexity(corr)
        full_rank = full_metrics['effective_rank']
        
        mask = np.ones(n, dtype=bool)
        mask[asset_idx] = False
        reduced_corr = corr[np.ix_(mask, mask)]
        
        if reduced_corr.shape[0] < 2:
            return 0.0
        
        reduced_metrics = self._eigenvalue_complexity(reduced_corr)
        reduced_rank = reduced_metrics['effective_rank']
        
        expected_reduction = (n - 1) / n
        actual_reduction = reduced_rank / full_rank if full_rank > 0 else 0
        
        contribution = expected_reduction - actual_reduction
        return contribution
    
    def _concentration_contribution(self, corr: np.ndarray, asset_idx: int) -> float:
        """How much does this asset contribute to the dominant correlation mode?"""
        corr = self._nearest_valid_corr(corr)
        eigenvalues, eigenvectors = np.linalg.eigh(corr)
        
        idx = np.argmax(eigenvalues)
        dominant_eigvec = eigenvectors[:, idx]
        
        return dominant_eigvec[asset_idx] ** 2
    
    def compute_correlation_surface(self, returns: pd.DataFrame) -> np.ndarray:
        """Compute rolling correlation matrices. Returns np.ndarray for backward compat."""
        corrs = []
        for i in range(self.base_window, len(returns) + 1):
            window = self._adaptive_window(returns, i)
            start = max(0, i - window)
            window_returns = returns.iloc[start:i]
            corr = window_returns.corr().values
            corrs.append(corr)
        return np.stack(corrs)

    def compute_complexity_metrics(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute eigenvalue-based complexity metrics over time."""
        corrs = self.compute_correlation_surface(returns)
        dates = returns.index[self.window-1:]
        
        metrics = []
        for i, (corr, date) in enumerate(zip(corrs, dates)):
            eig_metrics = self._eigenvalue_complexity(corr)
            metrics.append({
                'date': date,
                'spectral_entropy': eig_metrics['spectral_entropy'],
                'participation_ratio': eig_metrics['participation_ratio'],
                'effective_rank': eig_metrics['effective_rank'],
                'concentration': eig_metrics['concentration'],
                'eigenvalue_spread': eig_metrics['eigenvalue_spread']
            })
        
        return pd.DataFrame(metrics).set_index('date')

    def compute_etf_contributions(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Single-window contributions using structural importance metrics."""
        tickers = returns.columns.tolist()
        n_assets = len(tickers)
        if n_assets < 3:
            return pd.DataFrame()

        window = self._adaptive_window(returns, len(returns))
        recent_returns = returns.iloc[-window:]
        corr = recent_returns.corr().values
        corr = self._nearest_valid_corr(corr)
        
        centrality = self._eigenvector_centrality_contribution(corr)
        concentration_contrib = np.array([
            self._concentration_contribution(corr, i) for i in range(n_assets)
        ])
        
        marginal_contrib = np.array([
            self._marginal_complexity_contribution(corr, i) for i in range(n_assets)
        ])
        
        composites = []
        for i in range(n_assets):
            composite = (
                config.WEIGHT_LZ * (0.5 - centrality[i]) +
                config.WEIGHT_SAMPEN * (1 - concentration_contrib[i]) +
                config.WEIGHT_TSALLIS * marginal_contrib[i]
            )
            composites.append(composite)
        
        contributions = []
        for i, ticker in enumerate(tickers):
            contributions.append({
                'ticker': ticker,
                'contrib_lz': (0.5 - centrality[i]) if np.isfinite(centrality[i]) else 0.0,
                'contrib_samp': (1 - concentration_contrib[i]) if np.isfinite(concentration_contrib[i]) else 0.0,
                'contrib_tsallis': marginal_contrib[i] if np.isfinite(marginal_contrib[i]) else 0.0,
                'composite': composites[i] if np.isfinite(composites[i]) else 0.0
            })
        
        return pd.DataFrame(contributions)

    def compute_daily_avg_contributions(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Average structural contributions over sampled windows."""
        tickers = returns.columns.tolist()
        n_assets = len(tickers)
        if n_assets < 3:
            return pd.DataFrame()

        n = len(returns)
        step = max(1, (n - self.base_window) // 40)
        start_idx = self.base_window - 1
        idxs = list(range(start_idx, n, step))

        sum_centrality = np.zeros(n_assets)
        sum_concentration = np.zeros(n_assets)
        sum_marginal = np.zeros(n_assets)
        valid = 0

        for idx in idxs:
            window = self._adaptive_window(returns, idx + 1)
            start = max(0, idx + 1 - window)
            window_returns = returns.iloc[start:idx + 1]
            corr = window_returns.corr().values
            corr = self._nearest_valid_corr(corr)
            
            centrality = self._eigenvector_centrality_contribution(corr)
            concentration_contrib = np.array([
                self._concentration_contribution(corr, i) for i in range(n_assets)
            ])
            marginal_contrib = np.array([
                self._marginal_complexity_contribution(corr, i) for i in range(n_assets)
            ])
            
            sum_centrality += centrality
            sum_concentration += concentration_contrib
            sum_marginal += marginal_contrib
            valid += 1

        contributions = []
        for i, ticker in enumerate(tickers):
            avg_centrality = sum_centrality[i] / valid if valid > 0 else 0.5
            avg_concentration = sum_concentration[i] / valid if valid > 0 else 0.0
            avg_marginal = sum_marginal[i] / valid if valid > 0 else 0.0
            
            composite = (
                config.WEIGHT_LZ * (0.5 - avg_centrality) +
                config.WEIGHT_SAMPEN * (1 - avg_concentration) +
                config.WEIGHT_TSALLIS * avg_marginal
            )
            
            contributions.append({
                'ticker': ticker,
                'contrib_lz': (0.5 - avg_centrality) if np.isfinite(avg_centrality) else 0.0,
                'contrib_samp': (1 - avg_concentration) if np.isfinite(avg_concentration) else 0.0,
                'contrib_tsallis': avg_marginal if np.isfinite(avg_marginal) else 0.0,
                'composite': composite if np.isfinite(composite) else 0.0
            })
        
        return pd.DataFrame(contributions)

    def compute_global_contributions(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Averaged contributions over full history (sampled)."""
        return self.compute_daily_avg_contributions(returns)

    def compute_expected_return(self, returns: pd.DataFrame) -> pd.Series:
        """Expected return with momentum blend."""
        exp_ret = {}
        for ticker in returns.columns:
            ret = returns[ticker]
            if len(ret) >= 63:
                mom_short = ret.iloc[-21:].mean() * 252
                mom_medium = ret.iloc[-63:].mean() * 252
                blended = 0.6 * mom_short + 0.4 * mom_medium
                exp_ret[ticker] = blended
            elif len(ret) >= 21:
                exp_ret[ticker] = ret.iloc[-21:].mean() * 252
            else:
                exp_ret[ticker] = 0.0
        return pd.Series(exp_ret)

    def compute_global_expected_return(self, returns: pd.DataFrame) -> pd.Series:
        exp_ret = {}
        for ticker in returns.columns:
            ret = returns[ticker]
            exp_ret[ticker] = ret.mean() * 252
        return pd.Series(exp_ret)

    def compute_complexity_adjusted_return(self, expected_return: pd.Series,
                                           contributions: pd.DataFrame) -> pd.Series:
        """Adjust expected returns by complexity contribution (non-linear)."""
        contrib_map = contributions.set_index('ticker')['composite'].to_dict()
        
        contribs = contributions['composite'].values
        if len(contribs) > 0:
            contrib_spread = np.std(contribs)
        else:
            contrib_spread = 0
        
        adj_return = {}
        for ticker, exp in expected_return.items():
            contrib = contrib_map.get(ticker, 0.0)
            contrib = np.clip(contrib, -1, 1) if np.isfinite(contrib) else 0.0
            
            adjustment_strength = 0.3 + 0.4 * min(contrib_spread / 0.3, 1.0)
            adjustment = np.tanh(contrib * 2) * adjustment_strength
            
            adj_return[ticker] = exp * (1 + adjustment)
        
        return pd.Series(adj_return)
