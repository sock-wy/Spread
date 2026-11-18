"""Turn top-of-book data for two instruments into one aligned mid/spread table."""
from __future__ import annotations

import pandas as pd


def to_mid(df: pd.DataFrame, every: str = "1s") -> pd.DataFrame:
    """Last top-of-book state per interval -> mid and quoted spread.

    df: index = timestamp, columns bid_px_00, ask_px_00 (Databento mbp naming).
    """
    q = df[["bid_px_00", "ask_px_00"]].resample(every).last().ffill()
    return pd.DataFrame({"mid": (q["bid_px_00"] + q["ask_px_00"]) / 2,
                         "spread": q["ask_px_00"] - q["bid_px_00"]})


def align(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Columns: mid_a, spread_a, mid_b, spread_b on the common clock."""
    out = a.add_suffix("_a").join(b.add_suffix("_b"), how="inner").dropna()
    return out[(out["spread_a"] > 0) & (out["spread_b"] > 0)]
