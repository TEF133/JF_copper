"""
overfitting.py
==============
Overfitting diagnostics for the Strategies Lab. Two standard, complementary
checks (Bailey & López de Prado), computed from the grid of per-config daily
return series:

  • PBO  — Probability of Backtest Overfitting, via Combinatorially Symmetric
           Cross-Validation (CSCV). The fraction of IS/OOS folds in which the
           config that looks best in-sample lands below the median out-of-sample.
           PBO ≈ 0.5 ⇒ selection is no better than chance (overfit); PBO small
           ⇒ the in-sample winner generalises.
  • DSR  — Deflated Sharpe Ratio. The probability the selected strategy's true
           Sharpe is > 0 after deflating for the number of trials (grid size),
           sample length, skew and kurtosis. DSR > 0.95 ⇒ confident.

No scipy: the normal CDF / inverse-CDF come from the stdlib NormalDist.
"""
from __future__ import annotations

from itertools import combinations
from math import e, sqrt

import numpy as np
import pandas as pd
from statistics import NormalDist

_N = NormalDist()
_EULER = 0.5772156649015329


def _sr(r: np.ndarray) -> float:
    """Per-period Sharpe (mean/std); NaN if degenerate."""
    r = r[~np.isnan(r)]
    if r.size < 10 or r.std(ddof=1) == 0:
        return np.nan
    return r.mean() / r.std(ddof=1)


# ── PBO via CSCV ──────────────────────────────────────────────────────────────
def pbo_cscv(matrix: pd.DataFrame, s: int = 10) -> dict:
    """matrix: (time × config) daily returns. Split time into `s` blocks; over all
    C(s, s/2) IS/OOS partitions, find the IS-best config and rank it OOS."""
    M = matrix.dropna(how="all").to_numpy()
    T, N = M.shape
    if N < 2 or T < 2 * s:
        return {"pbo": np.nan, "lambdas": np.array([]), "n_trials": N, "folds": 0}
    bounds = np.linspace(0, T, s + 1).astype(int)
    blocks = [np.arange(bounds[i], bounds[i + 1]) for i in range(s)]
    lambdas = []
    for is_sel in combinations(range(s), s // 2):
        is_rows = np.concatenate([blocks[i] for i in is_sel])
        oos_rows = np.concatenate([blocks[i] for i in range(s) if i not in is_sel])
        sr_is = np.array([_sr(M[is_rows, c]) for c in range(N)])
        sr_oos = np.array([_sr(M[oos_rows, c]) for c in range(N)])
        if np.all(np.isnan(sr_is)):
            continue
        best = int(np.nanargmax(sr_is))
        valid = ~np.isnan(sr_oos)
        if not valid[best] or valid.sum() < 2:
            continue
        # relative OOS rank of the IS-best (1 = worst … n = best)
        order = sr_oos[valid]
        rank = (order < sr_oos[best]).sum() + 1
        omega = rank / (valid.sum() + 1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        lambdas.append(np.log(omega / (1 - omega)))
    lambdas = np.array(lambdas)
    pbo = float((lambdas <= 0).mean()) if lambdas.size else np.nan
    return {"pbo": pbo, "lambdas": lambdas, "n_trials": N, "folds": int(lambdas.size)}


# ── Deflated Sharpe Ratio ─────────────────────────────────────────────────────
def _psr(sr: float, T: int, skew: float, kurt: float, sr_star: float) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > sr_star). All SRs per-period."""
    denom = sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    return _N.cdf((sr - sr_star) * sqrt(T - 1) / denom)


def deflated_sharpe(best_returns: pd.Series, trial_sharpes: np.ndarray) -> dict:
    """DSR for the selected config given the spread of per-period Sharpes across
    the N trials. Returns dsr (prob true SR>0 after deflation) + the components."""
    r = best_returns.dropna().to_numpy()
    T = r.size
    if T < 20:
        return {"dsr": np.nan, "sr_ann": np.nan, "sr_star_ann": np.nan, "n_trials": int(trial_sharpes.size)}
    sr = _sr(r)
    s = pd.Series(r)
    skew, kurt = float(s.skew()), float(s.kurtosis() + 3.0)   # kurtosis (not excess)
    sr_trials = trial_sharpes[~np.isnan(trial_sharpes)]
    N = max(sr_trials.size, 1)
    var_sr = float(np.var(sr_trials, ddof=1)) if sr_trials.size > 1 else 0.0
    if var_sr <= 0 or N < 2:
        sr_star = 0.0
    else:
        z1 = _N.inv_cdf(1 - 1.0 / N)
        z2 = _N.inv_cdf(1 - 1.0 / (N * e))
        sr_star = sqrt(var_sr) * ((1 - _EULER) * z1 + _EULER * z2)
    dsr = _psr(sr, T, skew, kurt, sr_star)
    ann = sqrt(252)
    return {"dsr": float(dsr), "sr_ann": sr * ann, "sr_star_ann": sr_star * ann,
            "n_trials": int(N), "skew": skew, "kurt": kurt}


def verdict(pbo: float, dsr: float) -> tuple[str, str]:
    """(emoji, words) overall read of the two diagnostics."""
    if np.isnan(pbo) or np.isnan(dsr):
        return "⚪", "insufficient data"
    if pbo <= 0.3 and dsr >= 0.90:
        return "🟢", "robust — selection generalises, edge survives deflation"
    if pbo >= 0.6 or dsr <= 0.5:
        return "🔴", "likely overfit — discount this result"
    return "🟡", "mixed — treat with caution"
