"""Order-book features on a snapshot table.

Input: one row per book update (or per sampled instant), indexed by timestamp,
with columns bid_px_00, bid_sz_00, ask_px_00, ask_sz_00, ... (Databento mbp-10
naming). Level 00 is the best price.

Features
- mid, spread
- imbalance_l1: (bid_sz - ask_sz) / (bid_sz + ask_sz) at the best level
- imbalance_depth(n): same, summed over the first n levels
- microprice: size-weighted mid, leans toward the thinner side
- ofi: order-flow imbalance at the best level (Cont, Kukanov & Stoikov 2014)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def basic(df: pd.DataFrame) -> pd.DataFrame:
    bp, bs, ap, as_ = df["bid_px_00"], df["bid_sz_00"], df["ask_px_00"], df["ask_sz_00"]
    out = pd.DataFrame(index=df.index)
    out["mid"] = (bp + ap) / 2
    out["spread"] = ap - bp
    out["imbalance_l1"] = (bs - as_) / (bs + as_)
    out["microprice"] = (bp * as_ + ap * bs) / (bs + as_)
    return out


def imbalance_depth(df: pd.DataFrame, n: int = 5) -> pd.Series:
    b = sum(df[f"bid_sz_{i:02d}"] for i in range(n))
    a = sum(df[f"ask_sz_{i:02d}"] for i in range(n))
    return ((b - a) / (b + a)).rename(f"imbalance_d{n}")


def ofi(df: pd.DataFrame) -> pd.Series:
    """Per-update OFI e_t. Positive = net buying pressure at the touch.

    e_t = 1{Pb_t >= Pb_t-1} qb_t - 1{Pb_t <= Pb_t-1} qb_t-1
        - 1{Pa_t <= Pa_t-1} qa_t + 1{Pa_t >= Pa_t-1} qa_t-1
    """
    pb, qb, pa, qa = (df[c].to_numpy(float) for c in ("bid_px_00", "bid_sz_00", "ask_px_00", "ask_sz_00"))
    pb0, qb0, pa0, qa0 = (np.roll(x, 1) for x in (pb, qb, pa, qa))
    e = (pb >= pb0) * qb - (pb <= pb0) * qb0 - (pa <= pa0) * qa + (pa >= pa0) * qa0
    e[0] = 0.0
    return pd.Series(e, index=df.index, name="ofi")


def all_features(df: pd.DataFrame, depth_n: int = 5) -> pd.DataFrame:
    f = basic(df)
    if f"bid_sz_{depth_n - 1:02d}" in df:
        f[f"imbalance_d{depth_n}"] = imbalance_depth(df, depth_n)
    f["ofi"] = ofi(df)
    return f
