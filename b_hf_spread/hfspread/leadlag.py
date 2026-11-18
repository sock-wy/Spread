"""Who moves first? Cross-correlation of mid returns at different lags.

corr(r_a[t], r_b[t + k]) for k in [-K, K]:
- peak at k > 0  -> A leads B by k samples
- peak at k < 0  -> B leads A
- peak at k = 0  -> they move together at this sampling frequency (sample faster)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cross_corr(mid_a: pd.Series, mid_b: pd.Series, max_lag: int = 10) -> pd.Series:
    ra, rb = np.log(mid_a).diff(), np.log(mid_b).diff()
    out = {k: ra.corr(rb.shift(-k)) for k in range(-max_lag, max_lag + 1)}
    return pd.Series(out, name="corr").rename_axis("lag")


def lead_summary(cc: pd.Series) -> str:
    k = int(cc.drop(0).abs().idxmax())
    who = "A leads B" if k > 0 else "B leads A"
    return f"lag0 corr {cc.loc[0]:.3f}; strongest non-zero lag {k:+d} ({who}), corr {cc.loc[k]:.3f}"
