"""Two co-moving 1-second mid series where A leads B by `lag` seconds. Tests only."""
import numpy as np
import pandas as pd


def synthetic_pair(n_days: int = 3, lag: int = 2, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2025-03-03", periods=n_days)
    idx = pd.DatetimeIndex(np.concatenate([
        pd.date_range(d + pd.Timedelta(hours=9, minutes=30), d + pd.Timedelta(hours=16), freq="1s", inclusive="right")
        for d in days]))
    n = len(idx)
    common = rng.normal(0, 4e-5, n + lag)
    x = np.zeros(n)                                   # mean-reverting relative value
    for i in range(1, n):
        x[i] = 0.999 * x[i - 1] + rng.normal(0, 2e-5)
    ra = common[lag:]
    rb = common[:n] + rng.normal(0, 1e-5, n)          # B echoes A's move `lag` seconds later
    mid_a = 6000 * np.exp(np.cumsum(ra) + x)
    mid_b = 21000 * np.exp(np.cumsum(rb))
    return pd.DataFrame({"mid_a": mid_a, "spread_a": 0.25, "mid_b": mid_b, "spread_b": 0.25}, index=idx)
