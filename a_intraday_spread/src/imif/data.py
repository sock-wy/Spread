"""Data access: RiceQuant minute bars, CSV files, and a synthetic generator.

The original work pulled 1-minute bars for the non-adjusted continuous
contracts IM88 and IF88 from RiceQuant (rqdatac). Non-adjusted prices were
chosen over back-adjusted ones (IM888) because they reflect real tradable
levels for both legs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OHLC = ["open", "high", "low", "close"]


def load_rqdata(start: str, end: str, im: str = "IM88", if_: str = "IF88") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch 1-minute bars from RiceQuant. Requires a licensed rqdatac install."""
    import rqdatac  # noqa: WPS433 (optional dependency)

    rqdatac.init()
    out = []
    for code in (im, if_):
        df = rqdatac.get_price(code, start_date=start, end_date=end, frequency="1m", fields=OHLC)
        df = df.reset_index().set_index("datetime")[OHLC].sort_index()
        out.append(df)
    return out[0], out[1]


def load_csv(path_im: str | Path, path_if: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load two CSV files with columns datetime, open, high, low, close."""
    legs = []
    for p in (path_im, path_if):
        df = pd.read_csv(p, parse_dates=["datetime"]).set_index("datetime").sort_index()
        legs.append(df[OHLC].astype(float))
    return legs[0], legs[1]


def align_legs(im: pd.DataFrame, if_: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only timestamps present in both legs and drop incomplete bars."""
    idx = im.index.intersection(if_.index)
    im, if_ = im.loc[idx, OHLC].dropna(), if_.loc[idx, OHLC].dropna()
    idx = im.index.intersection(if_.index)
    return im.loc[idx], if_.loc[idx]


def validate_bars(df: pd.DataFrame, name: str = "") -> list[str]:
    """Basic sanity checks run before any backtest (mirrors the manual price check)."""
    issues = []
    if not df.index.is_monotonic_increasing:
        issues.append(f"{name}: index not sorted")
    if df.index.duplicated().any():
        issues.append(f"{name}: duplicated timestamps")
    bad = (df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))
    if bad.any():
        issues.append(f"{name}: {int(bad.sum())} bars with high/low outside open/close")
    if (df[OHLC] <= 0).any().any():
        issues.append(f"{name}: non-positive prices")
    return issues


def filter_rth(df: pd.DataFrame, start: str = "09:30", end: str = "16:00") -> pd.DataFrame:
    """Keep US regular trading hours. Timestamps must be bar-end times in America/New_York.

    CME futures trade ~23h a day; the strategy only trades the cash session, and
    the 17:00-18:00 ET maintenance break would otherwise look like a session gap.
    """
    t = df.index.time
    s, e = pd.Timestamp(start).time(), pd.Timestamp(end).time()
    return df[(t > s) & (t <= e)]


def _session_minutes_us(day: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(day + pd.Timedelta(hours=9, minutes=31), day + pd.Timedelta(hours=16), freq="1min")


def _session_minutes(day: pd.Timestamp) -> pd.DatetimeIndex:
    am = pd.date_range(day + pd.Timedelta(hours=9, minutes=31), day + pd.Timedelta(hours=11, minutes=30), freq="1min")
    pm = pd.date_range(day + pd.Timedelta(hours=13, minutes=1), day + pd.Timedelta(hours=15), freq="1min")
    return am.append(pm)


def synthetic_legs(
    n_days: int = 120,
    seed: int = 7,
    im0: float = 6000.0,
    if0: float = 3900.0,
    beta: float = 1.15,
    vol_if: float = 0.00045,
    kappa: float = 0.004,
    vol_spread: float = 0.00025,
    start: str = "2023-01-03",
    session: str = "cn",
    tick: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate two co-moving minute-bar legs whose log spread is mean-reverting.

    IF follows a random walk. IM = beta * IF move + an Ornstein-Uhlenbeck
    relative-strength term, which gives the strategy something to trade.
    This is a demo dataset only; it is not market data.
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_days)
    stamps = days.map(_session_minutes_us if session == "us" else _session_minutes)
    index = stamps[0].append(list(stamps[1:])) if len(stamps) > 1 else stamps[0]
    n = len(index)
    sub = 4  # sub-steps per bar to build OHLC

    r_if = rng.normal(0, vol_if / np.sqrt(sub), size=(n, sub))
    x = np.zeros((n, sub))
    xv = 0.0
    eps = rng.normal(0, vol_spread / np.sqrt(sub), size=(n, sub))
    for i in range(n):
        for j in range(sub):
            xv += -kappa / sub * xv + eps[i, j]
            x[i, j] = xv
    lif = np.log(if0) + np.cumsum(r_if.ravel()).reshape(n, sub)
    lim = np.log(im0) + beta * (lif - np.log(if0)) + x

    def bars(logp: np.ndarray) -> pd.DataFrame:
        p = np.exp(logp)
        prev_close = np.concatenate([[p[0, 0]], p[:-1, -1]])
        o = prev_close
        c = p[:, -1]
        h = np.maximum(p.max(axis=1), o)
        lo = np.minimum(p.min(axis=1), o)
        r = lambda a: np.round(a / tick) * tick  # noqa: E731 (tick rounding)
        df = pd.DataFrame({"open": r(o), "high": r(h), "low": r(lo), "close": r(c)}, index=index)
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)
        return df

    out_im, out_if = bars(lim), bars(lif)
    out_im.index.name = out_if.index.name = "datetime"
    return out_im, out_if


def resample_bars(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Aggregate 1-minute bar-end bars to N-minute bars (bar-end labels, bins from 09:30 ET)."""
    if minutes == 1:
        return df
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df:
        agg["volume"] = "sum"
    out = df.resample(f"{minutes}min", closed="right", label="right", offset="9h30min").agg(agg)
    return out.dropna(subset=["open", "close"])
