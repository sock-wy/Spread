"""Entry and exit rules on the spread.

Long the spread  = bet IM strengthens vs IF = buy IM + sell IF.
Short the spread = bet IF strengthens vs IM = buy IF + sell IM.

Layering: CCI pulls the trigger, Donchian confirms direction, EMA filters
counter-trend signals and manages the exit, RSI gates momentum and drives
take-profit. Rules use bar t (current, closed) and bar t-1 (prior).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def entry_signals(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Vectorised entry conditions. Returns boolean columns long_entry / short_entry."""
    c, o, h, lo = df["close"], df["open"], df["high"], df["low"]
    cci_now, cci_prev = df["cci"], df["cci"].shift(1)
    ema_now, ema_prev = df["ema"], df["ema"].shift(1)
    t = pd.Series(df.index.time, index=df.index)

    # critical_value gate: only trade once the spread has moved "far enough"
    # from its equilibrium centre (in either direction).
    dev = (c - df["center"]).abs()
    if cfg.use_critical_value:
        far_above = far_below = dev > cfg.critical_value_bp
    else:
        far_above = far_below = pd.Series(True, index=df.index)

    # Short spread (limit order, before 14:00):
    # prior bar broke the CCI low + current low falls;
    # filters: MA falling, no CCI reversal now, RSI > 20, close not too far below MA.
    short = (
        (cci_prev < cfg.cci_low)
        & (lo < lo.shift(1))
        & (ema_now < ema_prev)
        & (cci_now <= cci_prev)
        & (df["rsi"] > cfg.rsi_short_min)
        & ((ema_now - c).between(0.0, cfg.max_close_to_ema_bp))
        & far_above
        & (t < cfg.short_cutoff)
    )

    # Long spread (market order, before 14:15):
    # prior bar broke the Donchian upper band + no clear CCI reversal now;
    # filters: high breaks out, MA below close, bullish bar, RSI > 80.
    long_ = (
        (h.shift(1) > df["don_up"].shift(1))
        & (cci_now >= cci_prev)
        & (h > h.shift(1))
        & (ema_now < c)
        & (c > o)
        & (df["rsi"] > cfg.rsi_long_min)
        & far_below
        & (t < cfg.long_cutoff)
    )

    out = pd.DataFrame(index=df.index)
    out["short_entry"] = short.fillna(False) & cfg.enable_short
    out["long_entry"] = long_.fillna(False) & cfg.enable_long
    return out


def exit_reason(row_prev, row, side: int, entry_px: float, cfg) -> str | None:
    """Return the exit reason for an open position on the current bar, or None.

    side = +1 long spread, -1 short spread. Prices in bp of the spread.
    """
    c = row["close"]
    # --- stop-losses ---
    if side * (c - entry_px) <= -cfg.hard_stop_bp:
        return "hard_stop"
    if not np.isnan(row["ema"]) and side * (c - row["ema"]) < 0:
        return "ema_exit"
    if side > 0 and not np.isnan(row["don_lo"]) and row["low"] <= row["don_lo"]:
        return "opposite_band"
    if side < 0 and not np.isnan(row["don_up"]) and row["high"] >= row["don_up"]:
        return "opposite_band"
    if side < 0 and row["cci"] > cfg.cci_ceiling:
        return "cci_ceiling"
    # --- take-profits ---
    if side > 0 and row["rsi"] > cfg.rsi_overbought and row["cci"] > cfg.cci_high:
        return "rsi_cci_overbought"
    if side < 0 and row["rsi"] < cfg.rsi_oversold:
        return "rsi_oversold"
    # Prior bar CCI oversold, current bar back to normal -> anticipated rebound.
    # Only active in ranging conditions: suppressed when CCI fell below the floor.
    if (
        side < 0
        and row_prev["cci"] < cfg.cci_exit_low
        and row_prev["cci"] > cfg.cci_exit_floor
        and row["cci"] >= cfg.cci_exit_low
    ):
        return "cci_rebound"
    return None
