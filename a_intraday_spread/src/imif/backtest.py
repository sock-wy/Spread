"""Bar-by-bar backtest that executes spread signals as two futures legs.

Signals are evaluated on the spread at the close of bar t and filled on bar
t+1 (no look-ahead):

* market orders (long entry, exits) fill at the next bar's leg opens;
* the short entry is a limit order at the signal bar's spread close. It
  fills on the next bar if the spread trades at or above that level, and is
  cancelled otherwise;
* open positions are flattened at ``flat_time`` and before any session gap
  longer than the lunch break, so nothing is held overnight.

Both legs always enter and exit together, which is the property the
original two-strategy MultiCharts architecture was built to guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .indicators import add_indicators
from .metrics import summarize
from .signals import entry_signals, exit_reason
from .spread import build_spread


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    stats: dict
    frame: pd.DataFrame


def _fill(price: float, side: int, cost, leg: str = "im") -> float:
    """Apply slippage against us: buys fill higher, sells fill lower."""
    tick = cost.tick if leg == "im" or cost.tick_if is None else cost.tick_if
    return price + side * cost.slippage_ticks * tick


def _lots(px_im: float, px_if: float, cost) -> tuple[int, int]:
    ntl_im, ntl_if = px_im * cost.mult_im, px_if * cost.mult_if
    if cost.target_notional is None:
        return 1, max(1, int(round(ntl_im / ntl_if)))
    return max(1, int(round(cost.target_notional / ntl_im))), max(1, int(round(cost.target_notional / ntl_if)))


def run_backtest(im: pd.DataFrame, if_: pd.DataFrame, cfg: StrategyConfig | None = None) -> BacktestResult:
    cfg = cfg or StrategyConfig()
    cost = cfg.costs
    spread = build_spread(im, if_)
    df = add_indicators(spread, cfg)
    sig = entry_signals(df, cfg)
    df = df.join(sig)

    idx = df.index
    n = len(idx)
    times = idx.time
    dates = idx.normalize()
    gap_next = np.append((idx[1:] - idx[:-1]) > pd.Timedelta(hours=2), True)

    im_o, im_c = im["open"].to_numpy(), im["close"].to_numpy()
    if_o, if_c = if_["open"].to_numpy(), if_["close"].to_numpy()
    s_close, s_high = df["close"].to_numpy(), df["high"].to_numpy()
    s_open = df["open"].to_numpy()
    long_e, short_e = df["long_entry"].to_numpy(), df["short_entry"].to_numpy()

    pos = 0
    entry = {}
    pending = None  # ("long"|"short", limit_price or None)
    pending_exit = None
    trades = []
    realized = 0.0
    equity = np.empty(n)

    def open_trade(i: int, side: int, kind: str):
        n_im, n_if = _lots(im_o[i], if_o[i], cost)
        # long spread buys IM / sells IF; short spread the reverse
        px_im = _fill(im_o[i], side, cost, "im")
        px_if = _fill(if_o[i], -side, cost, "if")
        fee = (cost.open_rate * (px_im * cost.mult_im * n_im + px_if * cost.mult_if * n_if)
               + cost.fee_per_contract * (n_im + n_if))
        return {
            "side": side, "kind": kind, "i": i, "time": idx[i], "n_im": n_im, "n_if": n_if,
            "px_im": px_im, "px_if": px_if, "spread": s_open[i], "fee": fee,
        }

    def close_trade(i: int, px_im_raw: float, px_if_raw: float, reason: str, s_px: float):
        nonlocal realized
        e = entry
        side = e["side"]
        px_im = _fill(px_im_raw, -side, cost, "im")
        px_if = _fill(px_if_raw, side, cost, "if")
        pnl_im = side * e["n_im"] * cost.mult_im * (px_im - e["px_im"])
        pnl_if = -side * e["n_if"] * cost.mult_if * (px_if - e["px_if"])
        fee_out = (cost.close_rate * (px_im * cost.mult_im * e["n_im"] + px_if * cost.mult_if * e["n_if"])
                   + cost.fee_per_contract * (e["n_im"] + e["n_if"]))
        tick_if = cost.tick if cost.tick_if is None else cost.tick_if
        slip = 2 * cost.slippage_ticks * (cost.tick * e["n_im"] * cost.mult_im + tick_if * e["n_if"] * cost.mult_if)
        net = pnl_im + pnl_if - e["fee"] - fee_out
        realized += net
        trades.append({
            "entry_time": e["time"], "exit_time": idx[i], "side": "long" if side > 0 else "short",
            "order": e["kind"], "reason": reason, "spread_in": e["spread"], "spread_out": s_px,
            "n_im": e["n_im"], "n_if": e["n_if"], "im_in": e["px_im"], "im_out": px_im,
            "if_in": e["px_if"], "if_out": px_if, "gross": pnl_im + pnl_if,
            "commission": e["fee"] + fee_out, "slippage": slip, "net": net,
            "minutes": (idx[i] - e["time"]).total_seconds() / 60.0,
        })

    rows = df[["close", "high", "low", "ema", "don_up", "don_lo", "cci", "rsi"]]
    rec = rows.to_dict("records")

    for i in range(n):
        # 1) fills scheduled from the previous bar
        if pending_exit is not None and pos != 0:
            close_trade(i, im_o[i], if_o[i], pending_exit, s_open[i])
            pos, entry, pending_exit = 0, {}, None
        if pending is not None and pos == 0:
            kind, limit = pending
            if kind == "long":
                entry = open_trade(i, +1, "market")
                pos = +1
            elif s_high[i] >= limit:
                entry = open_trade(i, -1, "limit")
                pos = -1
            pending = None

        # 2) end-of-session flattening at this bar's close
        if pos != 0 and (times[i] >= cfg.flat_time or gap_next[i]):
            close_trade(i, im_c[i], if_c[i], "session_flat", s_close[i])
            pos, entry = 0, {}

        # 3) evaluate exits / entries on the closed bar
        elif pos != 0 and i > 0:
            r = exit_reason(rec[i - 1], rec[i], pos, entry["spread"], cfg)
            if r and i + 1 < n:
                pending_exit = r
        elif pos == 0 and i + 1 < n and not gap_next[i] and times[i] < cfg.flat_time:
            if long_e[i]:
                pending = ("long", None)
            elif short_e[i]:
                pending = ("short", s_close[i])

        # mark to market
        if pos != 0:
            e = entry
            mtm = pos * e["n_im"] * cost.mult_im * (im_c[i] - e["px_im"]) - pos * e["n_if"] * cost.mult_if * (if_c[i] - e["px_if"]) - e["fee"]
        else:
            mtm = 0.0
        equity[i] = cost.capital + realized + mtm

    trades_df = pd.DataFrame(trades)
    eq = pd.Series(equity, index=idx, name="equity")
    stats = summarize(trades_df, eq, cost.capital, bars_per_day=n / max(1, len(np.unique(dates))))
    return BacktestResult(trades=trades_df, equity=eq, stats=stats, frame=df)
