"""Estimating critical_value: how far the spread must stray from equilibrium.

The internship used a manual method: six "entangled" segments (the two legs
intertwined, spread oscillating around its centre) were picked by eye on a
5-year chart, and the mean absolute deviation of the spread inside those
segments became critical_value.

The statistical alternatives below were the recommended follow-up and give
a reproducible number that can be re-estimated on a rolling basis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def manual_critical_value(spread: pd.Series, segments: list[tuple[str, str]]) -> float:
    """Mean absolute deviation from each segment's own centre, averaged over segments."""
    vals = []
    for start, end in segments:
        seg = spread.loc[start:end].dropna()
        if len(seg):
            vals.append(float((seg - seg.mean()).abs().mean()))
    if not vals:
        raise ValueError("no data inside the given segments")
    return float(np.mean(vals))


def rolling_std_band(spread: pd.Series, window: int = 240 * 20, k: float = 1.0) -> pd.Series:
    """k x rolling standard deviation of the spread around its rolling mean."""
    return k * spread.rolling(window, min_periods=window // 4).std()


def percentile_band(spread: pd.Series, window: int = 240 * 20, q: float = 0.8) -> pd.Series:
    """Rolling q-quantile of the absolute deviation from the rolling mean."""
    dev = (spread - spread.rolling(window, min_periods=window // 4).mean()).abs()
    return dev.rolling(window, min_periods=window // 4).quantile(q)


def engle_granger(log_im: pd.Series, log_if: pd.Series) -> dict:
    """OLS hedge ratio of ln IM on ln IF and the residual's spread statistics.

    If statsmodels is installed an ADF test is run on the residual.
    """
    x = np.column_stack([np.ones(len(log_if)), log_if.to_numpy()])
    y = log_im.to_numpy()
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = pd.Series(y - x @ coef, index=log_im.index)
    out = {"alpha": float(coef[0]), "beta": float(coef[1]), "resid_std_bp": float(resid.std() * 1e4)}
    try:
        from statsmodels.tsa.stattools import adfuller

        stat, pvalue, *_ = adfuller(resid.to_numpy(), maxlag=20, autolag=None)
        out.update({"adf_stat": float(stat), "adf_pvalue": float(pvalue)})
    except ImportError:
        out["adf_stat"] = None
    # half-life of mean reversion from an AR(1) fit on the residual
    r = resid.to_numpy()
    b = np.polyfit(r[:-1], np.diff(r), 1)[0]
    out["half_life_bars"] = float(-np.log(2) / b) if b < 0 else float("inf")
    return out
