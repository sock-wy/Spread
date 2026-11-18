"""Baseline study: does book state predict the next mid-price move, and is it
big enough to beat the cost of crossing the spread?

1. Sample features on a fixed clock (default 1s) so busy periods don't dominate.
2. Target = mid change over the next `horizon` samples, in ticks.
3. Fit a one-variable linear model per feature (in-sample first 70%, test on last 30%).
4. Report out-of-sample R^2, sign hit rate, and the share of signals whose
   predicted move exceeds half the spread (the cost of a taker entry).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sample(feat: pd.DataFrame, every: str = "1s") -> pd.DataFrame:
    """Last state in each interval; OFI is summed over the interval."""
    out = feat.drop(columns=["ofi"]).resample(every).last()
    out["ofi"] = feat["ofi"].resample(every).sum()
    return out.dropna(subset=["mid"])


def add_target(s: pd.DataFrame, horizon: int, tick: float) -> pd.DataFrame:
    s = s.copy()
    s["target_ticks"] = (s["mid"].shift(-horizon) - s["mid"]) / tick
    return s.dropna(subset=["target_ticks"])


def evaluate(s: pd.DataFrame, features: list[str], tick: float, train_frac: float = 0.7) -> pd.DataFrame:
    cut = int(len(s) * train_frac)
    tr, te = s.iloc[:cut], s.iloc[cut:]
    rows = []
    for f in features:
        x_tr, y_tr = tr[f].to_numpy(float), tr["target_ticks"].to_numpy(float)
        ok = np.isfinite(x_tr) & np.isfinite(y_tr)
        beta, alpha = np.polyfit(x_tr[ok], y_tr[ok], 1)
        x_te, y_te = te[f].to_numpy(float), te["target_ticks"].to_numpy(float)
        ok = np.isfinite(x_te) & np.isfinite(y_te)
        pred, y = alpha + beta * x_te[ok], y_te[ok]
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        moved = y != 0
        hit = np.mean(np.sign(pred[moved]) == np.sign(y[moved])) if moved.any() else np.nan
        half_spread = te["spread"].to_numpy(float)[ok] / tick / 2
        rows.append({"feature": f, "beta": beta, "oos_r2": r2, "hit_rate": hit,
                     "pct_pred_beats_half_spread": np.mean(np.abs(pred) > half_spread)})
    return pd.DataFrame(rows).set_index("feature")
