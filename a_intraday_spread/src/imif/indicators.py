"""Indicators computed on the spread series (never on a single leg)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def wilder_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI using his smoothing (alpha = 1/n)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    return rsi.where(avg_gain.notna())


def cci(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20) -> pd.Series:
    """Commodity Channel Index with the standard 0.015 constant."""
    tp = (high + low + close) / 3.0
    sma = tp.rolling(n).mean()
    mad = _rolling_mad(tp, n)
    return (tp - sma) / (0.015 * mad.replace(0.0, np.nan))


def _rolling_mad(s: pd.Series, n: int) -> pd.Series:
    """Rolling mean absolute deviation, vectorised (same result as rolling.apply)."""
    x = s.to_numpy(float)
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        w = np.lib.stride_tricks.sliding_window_view(x, n)
        out[n - 1:] = np.abs(w - w.mean(axis=1, keepdims=True)).mean(axis=1)
    return pd.Series(out, index=s.index)


def donchian(high: pd.Series, low: pd.Series, n: int = 20) -> tuple[pd.Series, pd.Series]:
    """Donchian channel of the previous n bars (excludes the current bar, so a
    breakout means the current bar exceeds the prior range)."""
    upper = high.rolling(n).max().shift(1)
    lower = low.rolling(n).min().shift(1)
    return upper, lower


def ema(close: pd.Series, n: int = 30) -> pd.Series:
    return close.ewm(span=n, adjust=False, min_periods=n).mean()


def add_indicators(spread: pd.DataFrame, cfg) -> pd.DataFrame:
    df = spread.copy()
    df["rsi"] = wilder_rsi(df["close"], cfg.len_rsi)
    df["cci"] = cci(df["high"], df["low"], df["close"], cfg.len_cci)
    df["don_up"], df["don_lo"] = donchian(df["high"], df["low"], cfg.len_don)
    df["ema"] = ema(df["close"], cfg.len_ema)
    df["center"] = df["close"].rolling(cfg.center_window, min_periods=cfg.center_window // 4).mean()
    return df
