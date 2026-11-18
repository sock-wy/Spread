"""Staged parameter search + holdout test + walk-forward check for the US pair.

    python scripts/tune_us.py --a data/IWM_1m.csv --b data/SPY_1m.csv --dev-end 2024-01-01

Design
- Every grid point is backtested ONCE on the full history. The strategy is flat
  every night, so a run's daily P&L can be sliced by date without re-running.
- Parameters are chosen on the development period only (before --dev-end), by
  plateau Sharpe (median Sharpe of the point and its grid neighbours), with a
  minimum trade count. The holdout period after --dev-end is never used to choose.
- Stages: 1 indicator lengths -> 2 entry thresholds -> 3 exits/risk. Each stage
  starts from the previous stage's pick.
- Walk-forward check on the stage-1 grid: re-pick every quarter using the
  previous 12 months, trade the next 3 months, stitch the out-of-sample days.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imif import StrategyConfig, run_backtest  # noqa: E402
from imif.data import align_legs, load_csv, resample_bars  # noqa: E402
from imif.indicators import add_indicators  # noqa: E402
from imif.optimize import plateau_scores  # noqa: E402
from imif.spread import build_spread  # noqa: E402

LEGS: tuple[pd.DataFrame, pd.DataFrame] | None = None
MIN_DEV_TRADES = 100


# ---------------------------------------------------------------- running
def _daily(eq: pd.Series, capital: float) -> pd.Series:
    d = eq.groupby(eq.index.normalize()).last()
    return d.diff().fillna(d.iloc[0] - capital)


def _run(cfg: StrategyConfig) -> tuple[pd.Series, pd.Series]:
    """Daily net P&L and daily trade count for one config on the full history."""
    res = run_backtest(*LEGS, cfg)
    pnl = _daily(res.equity, cfg.costs.capital)
    if len(res.trades):
        n = res.trades.groupby(pd.to_datetime(res.trades["exit_time"]).dt.normalize()).size()
        n = n.reindex(pnl.index, fill_value=0)
    else:
        n = pd.Series(0, index=pnl.index)
    return pnl, n


def run_grid(base: StrategyConfig, grid: dict, jobs: int) -> tuple[list[dict], list[pd.Series], list[pd.Series]]:
    keys = list(grid)
    combos = [dict(zip(keys, c)) for c in itertools.product(*(grid[k] for k in keys))]
    cfgs = [replace(base, **c) for c in combos]
    with Pool(jobs) as pool:
        out = pool.map(_run, cfgs, chunksize=1)
    return combos, [o[0] for o in out], [o[1] for o in out]


def sharpe(pnl: pd.Series, capital: float) -> float:
    r = pnl / capital
    return float(np.sqrt(252) * r.mean() / r.std()) if r.std() > 0 else float("nan")


def score(combos, pnls, counts, mask_fn, capital, keys) -> pd.DataFrame:
    rows = []
    for c, p, n in zip(combos, pnls, counts):
        m = mask_fn(p.index)
        tr = int(n[m].sum())
        s = sharpe(p[m], capital) if tr >= MIN_DEV_TRADES else float("nan")
        rows.append({**c, "sharpe": s, "trades": tr, "net": float(p[m].sum())})
    return plateau_scores(pd.DataFrame(rows), keys)


# ---------------------------------------------------------------- stats
def perf(pnl: pd.Series, capital: float) -> dict:
    eq = capital + pnl.cumsum()
    yrs = len(pnl) / 252
    return {"net": float(pnl.sum()), "ann_return": float((eq.iloc[-1] / capital) ** (1 / yrs) - 1) if yrs > 0 else np.nan,
            "sharpe": sharpe(pnl, capital), "max_dd": float((eq / eq.cummax() - 1).min()),
            "days": len(pnl)}


def trade_stats(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0}
    w, l = t.loc[t["net"] > 0, "net"].sum(), -t.loc[t["net"] < 0, "net"].sum()
    return {"trades": len(t), "win_rate": float((t["net"] > 0).mean()), "profit_factor": float(w / l) if l else np.inf,
            "avg_net": float(t["net"].mean()), "avg_minutes": float(t["minutes"].mean()),
            "long": int((t["side"] == "long").sum()), "short": int((t["side"] == "short").sum())}


# ---------------------------------------------------------------- main
def main() -> None:
    global LEGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="data/IWM_1m.csv"), ap.add_argument("--b", default="data/SPY_1m.csv")
    ap.add_argument("--preset", choices=["iwm_spy", "rty_es"], default="iwm_spy")
    ap.add_argument("--dev-end", default="2024-01-01")
    ap.add_argument("--out", default="output/tune_iwm_spy")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--timeframes", type=int, nargs="+", default=[1, 3, 5, 10, 15])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    legs_1m = align_legs(*load_csv(args.a, args.b))
    base = StrategyConfig.us_iwm_spy() if args.preset == "iwm_spy" else StrategyConfig.us_rty_es()
    cap = base.costs.capital
    dev_end = pd.Timestamp(args.dev_end)
    is_dev = lambda idx: idx < dev_end  # noqa: E731
    print(f"bars {len(legs_1m[0]):,}  {legs_1m[0].index[0]} -> {legs_1m[0].index[-1]}  dev < {dev_end.date()}")
    g1 = {"len_cci": [10, 14, 20, 26, 34], "len_don": [10, 15, 20, 30, 40], "len_ema": [15, 20, 30, 45, 60]}

    # Stage 0: bar size. The IM/IF original used 1-minute bars; whether that is
    # right for this pair (noise vs cost) is itself a parameter. For each bar size
    # run the stage-1 grid with the original thresholds and keep the best plateau.
    s0 = []
    for tf in args.timeframes:
        LEGS = tuple(resample_bars(x, tf) for x in legs_1m)
        b = replace(base, center_window=max(20, 390 // tf))
        combos, pnls, counts = run_grid(b, g1, args.jobs)
        res = score(combos, pnls, counts, is_dev, cap, list(g1))
        res.to_csv(out / f"stage0_tf{tf}.csv", index=False)
        top = res.dropna(subset=["plateau_sharpe", "sharpe"])
        s0.append({"tf": tf, "best_plateau_sharpe": float(top.iloc[0]["plateau_sharpe"]) if len(top) else np.nan,
                   "median_sharpe": float(res["sharpe"].median()), "trades_at_best": int(top.iloc[0]["trades"]) if len(top) else 0})
        print(f"stage0 tf={tf}m: best plateau {s0[-1]['best_plateau_sharpe']:.2f}, grid median {s0[-1]['median_sharpe']:.2f}")
    s0 = pd.DataFrame(s0)
    s0.to_csv(out / "stage0.csv", index=False)
    tf = int(s0.sort_values("best_plateau_sharpe", ascending=False).iloc[0]["tf"])
    LEGS = tuple(resample_bars(x, tf) for x in legs_1m)
    base = replace(base, center_window=max(20, 390 // tf))
    print(f"stage0: bar size {tf} min")

    # spread scale on the dev period -> bp grids
    sp = add_indicators(build_spread(*LEGS), base)
    dev_sp = sp[sp.index < dev_end]
    dev_abs = (dev_sp["close"] - dev_sp["center"]).abs().dropna()
    one_min = dev_sp["close"].diff().abs()
    scale = {"dev_q50": float(dev_abs.quantile(0.5)), "dev_q75": float(dev_abs.quantile(0.75)),
             "dev_q90": float(dev_abs.quantile(0.9)), "abs_bar_move_median": float(one_min.median()),
             "abs_bar_move_q90": float(one_min.quantile(0.9))}
    print("spread scale (bp):", {k: round(v, 2) for k, v in scale.items()})
    q50, q75, q90, m90 = scale["dev_q50"], scale["dev_q75"], scale["dev_q90"], scale["abs_bar_move_q90"]
    rnd = lambda xs: sorted({round(float(x), 1) for x in xs})  # noqa: E731

    log = {"scale_bp": scale, "dev_end": str(dev_end.date()), "bar_minutes": tf, "stage0": s0.to_dict("records"), "stages": {}}
    cfg = base

    # Stage 2: entry thresholds (bp grids scaled to this pair)
    g2 = {"cci_low": [-60.0, -80.0, -100.0, -130.0, -160.0], "rsi_long_min": [60.0, 70.0, 80.0],
          "rsi_short_min": [20.0, 30.0, 40.0], "critical_value_bp": rnd([0.5 * q50, q50, q75, q90])}
    # Stage 3: exits / risk
    g3 = {"hard_stop_bp": rnd([1.0 * q75, 1.5 * q75, 2.0 * q75, 3.0 * q75]),
          "max_close_to_ema_bp": rnd([1 * m90, 2 * m90, 4 * m90]),
          "cci_exit_low": [-100.0, -150.0, -200.0]}

    stage_runs = {}
    for name, grid in (("stage1", g1), ("stage2", g2), ("stage3", g3)):
        combos, pnls, counts = run_grid(cfg, grid, args.jobs)
        res = score(combos, pnls, counts, is_dev, cap, list(grid))
        res.to_csv(out / f"{name}.csv", index=False)
        pick = res.dropna(subset=["plateau_sharpe", "sharpe"]).iloc[0]   # the point itself must pass the trade floor
        chosen = {k: (int(pick[k]) if isinstance(grid[k][0], int) else float(pick[k])) for k in grid}
        cfg = replace(cfg, **chosen)
        stage_runs[name] = (combos, pnls, counts)
        log["stages"][name] = {"grid": grid, "chosen": chosen, "dev_sharpe": float(pick["sharpe"]),
                               "dev_plateau_sharpe": float(pick["plateau_sharpe"]), "dev_trades": int(pick["trades"])}
        print(f"{name}: {chosen}  dev sharpe {pick['sharpe']:.2f}  plateau {pick['plateau_sharpe']:.2f}  trades {int(pick['trades'])}")

    # Final + baseline + cost sensitivity on full history
    final = run_backtest(*LEGS, cfg)
    orig = StrategyConfig.us_iwm_spy() if args.preset == "iwm_spy" else StrategyConfig.us_rty_es()
    baseline = run_backtest(*legs_1m, orig)   # original IM/IF params on 1-minute bars
    pnl_f, pnl_b = _daily(final.equity, cap), _daily(baseline.equity, cap)
    dev_m, hold_m = pnl_f.index < dev_end, pnl_f.index >= dev_end
    tr = final.trades.copy()
    tr["exit_time"] = pd.to_datetime(tr["exit_time"])
    tr["entry_time"] = pd.to_datetime(tr["entry_time"])
    bt = baseline.trades.assign(exit_time=lambda d: pd.to_datetime(d["exit_time"]))
    tr_dev, tr_hold = tr[tr["exit_time"] < dev_end], tr[tr["exit_time"] >= dev_end]

    cost_rows = []
    for slip in (0.0, 0.5, 1.0, 2.0):
        c = replace(cfg, costs=replace(cfg.costs, slippage_ticks=slip))
        p = _daily(run_backtest(*LEGS, c).equity, cap)
        cost_rows.append({"slippage_ticks": slip, **{f"dev_{k}": v for k, v in perf(p[dev_m], cap).items()},
                          **{f"hold_{k}": v for k, v in perf(p[hold_m], cap).items()}})

    # Walk-forward on the stage-1 grid (thresholds/exits fixed at final picks)
    combos, pnls, counts = run_grid(cfg, g1, args.jobs)
    days = pnls[0].index
    wf, wf_picks = [], []
    t0 = days[0] + pd.DateOffset(months=12)
    while t0 < days[-1]:
        t1 = t0 + pd.DateOffset(months=3)
        tr_mask = lambda idx, a=t0: (idx >= a - pd.DateOffset(months=12)) & (idx < a)  # noqa: E731
        res = score(combos, pnls, counts, tr_mask, cap, list(g1)).dropna(subset=["plateau_sharpe", "sharpe"])
        if len(res):
            k = res.index[0]
            m = (days >= t0) & (days < t1)
            wf.append(pnls[k][m])
            wf_picks.append({"test_start": str(t0.date()), **combos[k],
                             "train_plateau_sharpe": float(res.iloc[0]["plateau_sharpe"])})
        t0 = t1
    pnl_wf = pd.concat(wf) if wf else pd.Series(dtype=float)

    # Save
    summary = {
        "final_params": {"bar_minutes": tf, "center_window": cfg.center_window,
                         **{k: getattr(cfg, k) for k in list(g1) + list(g2) + list(g3)}},
        "dev": {**perf(pnl_f[dev_m], cap), **trade_stats(tr_dev)},
        "holdout": {**perf(pnl_f[hold_m], cap), **trade_stats(tr_hold)},
        "baseline_cn_params_dev": {**perf(pnl_b[dev_m], cap), **trade_stats(bt[bt["exit_time"] < dev_end])},
        "baseline_cn_params_holdout": {**perf(pnl_b[hold_m], cap), **trade_stats(bt[bt["exit_time"] >= dev_end])},
        "walk_forward_oos": perf(pnl_wf, cap) if len(pnl_wf) else {},
        "walk_forward_note": "stage-1 lengths re-picked each quarter on the prior 12 months; thresholds/exits fixed at final picks",
    }
    log["summary"] = summary
    (out / "summary.json").write_text(json.dumps(log, indent=2, default=str))
    pd.DataFrame(cost_rows).to_csv(out / "cost_sensitivity.csv", index=False)
    pd.DataFrame(wf_picks).to_csv(out / "walk_forward_picks.csv", index=False)
    tr.to_csv(out / "trades_final.csv", index=False)
    pd.DataFrame({"final": pnl_f, "baseline_cn": pnl_b}).to_csv(out / "daily_pnl.csv")
    pnl_wf.rename("walk_forward").to_csv(out / "daily_pnl_walk_forward.csv")

    from tune_report import make_report  # noqa: E402
    make_report(out, cfg, StrategyConfig.us_iwm_spy() if args.preset == 'iwm_spy' else StrategyConfig.us_rty_es(), pnl_f, pnl_b, pnl_wf, tr, dev_end, cap, stage_runs, g1, g2)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
