"""Performance statistics, including the design targets from the internship."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Design goals for the IM/IF instrument (from the strategy summary)
TARGETS = {
    "win_rate": 0.40,          # > 40%
    "annual_return": 0.10,     # ~10%
    "avg_trade_net": 500.0,    # > 500 CNY per trade
    "sharpe": 1.0,             # > 1
    "avg_minutes": 50.0,       # ~50 minutes
    "time_in_market": (0.05, 0.10),
}


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float(((equity - peak) / peak).min())


def summarize(trades: pd.DataFrame, equity: pd.Series, capital: float) -> dict:
    daily = equity.groupby(equity.index.normalize()).last()
    rets = daily.pct_change().dropna()
    n_days = max(len(daily), 1)
    sharpe = float(np.sqrt(252) * rets.mean() / rets.std()) if rets.std() > 0 else float("nan")
    total = float(equity.iloc[-1] - capital)
    stats = {
        "net_profit": total,
        "annual_return": (equity.iloc[-1] / capital) ** (252 / n_days) - 1 if n_days > 1 else float("nan"),
        "max_drawdown": max_drawdown(equity),
        "sharpe": sharpe,
        "trades": int(len(trades)),
        "leg_trades": int(2 * len(trades)),  # each spread trade = one IM + one IF trade
    }
    if len(trades):
        wins = trades.loc[trades["net"] > 0, "net"].sum()
        losses = -trades.loc[trades["net"] < 0, "net"].sum()
        held = trades["minutes"].sum()
        session_minutes = n_days * 240
        stats.update({
            "win_rate": float((trades["net"] > 0).mean()),
            "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
            "avg_trade_net": float(trades["net"].mean()),
            "avg_minutes": float(trades["minutes"].mean()),
            "time_in_market": float(held / session_minutes),
            "commission_paid": float(trades["commission"].sum()),
            "slippage_paid": float(trades["slippage"].sum()),
            "long_trades": int((trades["side"] == "long").sum()),
            "short_trades": int((trades["side"] == "short").sum()),
        })
    return stats


def check_targets(stats: dict) -> pd.DataFrame:
    rows = []
    for k, goal in TARGETS.items():
        v = stats.get(k, float("nan"))
        if isinstance(goal, tuple):
            ok = goal[0] <= v <= goal[1]
            goal_s = f"{goal[0]:.0%}–{goal[1]:.0%}"
        elif k == "avg_minutes":
            ok = 0.5 * goal <= v <= 2 * goal
            goal_s = f"~{goal:.0f}"
        else:
            ok = v > goal
            goal_s = f"> {goal}"
        rows.append({"metric": k, "value": v, "target": goal_s, "met": bool(ok)})
    return pd.DataFrame(rows)
