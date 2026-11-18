"""Toy L1 book where order-flow imbalance genuinely leads the mid. Tests only."""
import numpy as np
import pandas as pd


def synthetic_book(n: int = 20_000, tick: float = 0.25, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.Timestamp("2025-03-03 09:30") + pd.to_timedelta(np.cumsum(rng.exponential(0.2, n)), unit="s")
    bid = 6000.0
    rows = []
    bs, as_ = 50.0, 50.0
    for _ in range(n):
        bs = max(1.0, bs + rng.normal(0, 8))
        as_ = max(1.0, as_ + rng.normal(0, 8))
        imb = (bs - as_) / (bs + as_)
        if rng.random() < 0.05 + 0.25 * abs(imb):          # price moves toward the thin side
            bid += tick * np.sign(imb)
            bs, as_ = 50.0, 50.0
        rows.append((bid, bs, bid + tick, as_))
    return pd.DataFrame(rows, index=idx, columns=["bid_px_00", "bid_sz_00", "ask_px_00", "ask_sz_00"])
