"""Baseline high-frequency pairs strategy on two futures (e.g. ES vs NQ, ES vs RTY).

Spread   s_t = ln(A_t) - beta_t * ln(B_t), beta from a rolling OLS (no look-ahead).
Signal   z_t = (s_t - mean) / std over a rolling window.
Rules    |z| > z_in  -> enter against the move (short spread if z > 0)
         |z| < z_out -> exit;  also exit after max_hold samples or at session end.
Costs    taker on both legs: pay half the quoted spread per leg per side + fee.

This is the plain, honest version to measure against. Improvements to try next:
OFI filter from c_orderbook, passive (maker) entries, Kalman-filter beta.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PairsConfig:
    beta_window: int = 3600       # samples (1h at 1s)
    z_window: int = 900           # samples (15 min at 1s)
    z_in: float = 2.5
    z_out: float = 0.5
    max_hold: int = 600           # samples
    mult_a: float = 50.0          # ES $50/pt
    mult_b: float = 20.0          # NQ $20/pt
    fee_per_contract: float = 2.25
    lots_a: int = 1


def rolling_beta(la: pd.Series, lb: pd.Series, window: int) -> pd.Series:
    cov = la.rolling(window).cov(lb)
    var = lb.rolling(window).var()
    return (cov / var).shift(1)          # use only past data


def zscore(q: pd.DataFrame, cfg: PairsConfig) -> pd.DataFrame:
    la, lb = np.log(q["mid_a"]), np.log(q["mid_b"])
    beta = rolling_beta(la, lb, cfg.beta_window)
    s = la - beta * lb
    m, sd = s.rolling(cfg.z_window).mean().shift(1), s.rolling(cfg.z_window).std().shift(1)
    return q.assign(beta=beta, spread_val=s, z=(s - m) / sd)


def backtest(q: pd.DataFrame, cfg: PairsConfig) -> tuple[pd.DataFrame, dict]:
    d = zscore(q, cfg).dropna(subset=["z", "beta"])
    z = d["z"].to_numpy()
    ma, mb = d["mid_a"].to_numpy(), d["mid_b"].to_numpy()
    sa, sb = d["spread_a"].to_numpy(), d["spread_b"].to_numpy()
    beta = d["beta"].to_numpy()
    day = d.index.normalize()
    last_of_day = np.append(day[1:] != day[:-1], True)

    trades, pos, i0 = [], 0, 0
    lots_b = 0
    for i in range(len(d)):
        if pos != 0 and (abs(z[i]) < cfg.z_out or i - i0 >= cfg.max_hold or last_of_day[i]):
            # exit: long spread = long A / short B
            pnl = pos * (cfg.lots_a * cfg.mult_a * (ma[i] - ma[i0]) - lots_b * cfg.mult_b * (mb[i] - mb[i0]))
            cost = (cfg.lots_a * cfg.mult_a * (sa[i0] + sa[i]) / 2 + lots_b * cfg.mult_b * (sb[i0] + sb[i]) / 2
                    + 2 * cfg.fee_per_contract * (cfg.lots_a + lots_b))
            trades.append({"entry": d.index[i0], "exit": d.index[i], "side": pos, "z_in": z[i0],
                           "lots_b": lots_b, "gross": pnl, "cost": cost, "net": pnl - cost,
                           "samples": i - i0})
            pos = 0
        elif pos == 0 and abs(z[i]) > cfg.z_in and not last_of_day[i]:
            pos, i0 = (-1 if z[i] > 0 else 1), i
            # hedge in dollar-beta terms: B notional = beta * A notional
            lots_b = max(1, int(round(beta[i] * cfg.lots_a * cfg.mult_a * ma[i] / (cfg.mult_b * mb[i]))))
    t = pd.DataFrame(trades)
    if t.empty:
        return t, {"trades": 0}
    stats = {"trades": len(t), "win_rate": float((t["net"] > 0).mean()),
             "gross_total": float(t["gross"].sum()), "cost_total": float(t["cost"].sum()),
             "net_total": float(t["net"].sum()), "avg_net": float(t["net"].mean()),
             "avg_hold_samples": float(t["samples"].mean())}
    return t, stats
