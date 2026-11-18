"""Synthetic spread instrument.

Three representations of IM/IF relative strength were compared:

1. Difference IM - IF: rejected, dominated by the price-level gap (~6000 vs ~3000).
2. Ratio IM / IF: rejected, not additive and asymmetric.
3. Cumulative log-return spread: selected.

    S_t = sum ln(IM_t / IM_{t-1}) - sum ln(IF_t / IF_{t-1})
        = ln(IM_t / IM_0) - ln(IF_t / IF_0)

The series is expressed in basis points (x 10,000) so thresholds read naturally.

Synthetic OHLC: the two legs' intraday extremes do not happen at the same
instant, so the spread's high/low are not computed from leg highs/lows.
Instead the spread is evaluated at the bar open and the bar close, and
high = max(open, close), low = min(open, close).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BP = 1e4


def cumulative_log_spread(im_close: pd.Series, if_close: pd.Series) -> pd.Series:
    r_im = np.log(im_close).diff().fillna(0.0)
    r_if = np.log(if_close).diff().fillna(0.0)
    return (r_im.cumsum() - r_if.cumsum()) * BP


def build_spread(im: pd.DataFrame, if_: pd.DataFrame) -> pd.DataFrame:
    """Return synthetic spread OHLC (bp) on the common index of the two legs."""
    base_im, base_if = im["close"].iloc[0], if_["close"].iloc[0]
    s_close = (np.log(im["close"] / base_im) - np.log(if_["close"] / base_if)) * BP
    s_open = (np.log(im["open"] / base_im) - np.log(if_["open"] / base_if)) * BP
    out = pd.DataFrame(
        {
            "open": s_open,
            "high": np.maximum(s_open, s_close),
            "low": np.minimum(s_open, s_close),
            "close": s_close,
        },
        index=im.index,
    )
    return out


def rejected_representations(im_close: pd.Series, if_close: pd.Series) -> pd.DataFrame:
    """The two alternatives that were tested and rejected, kept for comparison plots."""
    return pd.DataFrame({"difference": im_close - if_close, "ratio": im_close / if_close})
