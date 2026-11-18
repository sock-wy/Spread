"""Grid search with plateau scoring and an out-of-sample check.

    python scripts/optimize.py --source synthetic
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imif import StrategyConfig, run_backtest  # noqa: E402
from imif.data import align_legs, filter_rth, load_csv, synthetic_legs  # noqa: E402
from imif.optimize import grid_search, plateau_scores, walk_forward_split  # noqa: E402

GRID = {
    "len_cci": [14, 20, 26],
    "len_ema": [20, 30, 40],
    "len_don": [15, 20, 30],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "csv"], default="synthetic")
    ap.add_argument("--im"), ap.add_argument("--if", dest="if_")
    ap.add_argument("--out", default="output")
    ap.add_argument("--market", choices=["cn", "us"], default="us")
    args = ap.parse_args()

    base = StrategyConfig.us_rty_es() if args.market == "us" else StrategyConfig()
    if args.source == "synthetic":
        im, if_ = (synthetic_legs(n_days=120, session="us", im0=2400, if0=6000, tick=0.25)
                   if args.market == "us" else synthetic_legs(n_days=120))
    else:
        im, if_ = load_csv(args.im, args.if_)
        if args.market == "us":
            im, if_ = filter_rth(im), filter_rth(if_)
    im, if_ = align_legs(im, if_)
    (im_is, if_is), (im_os, if_os) = walk_forward_split(im, if_, 0.7)

    keys = list(GRID)
    res = plateau_scores(grid_search(im_is, if_is, GRID, base), keys)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "optimization.csv", index=False)

    best_point = res.sort_values("sharpe", ascending=False).iloc[0]
    best_plateau = res.iloc[0]
    print("Top by raw Sharpe (spike risk):\n", best_point[keys + ["sharpe", "plateau_sharpe", "trades"]].to_string())
    print("\nTop by plateau Sharpe (selected):\n", best_plateau[keys + ["sharpe", "plateau_sharpe", "trades"]].to_string())

    cfg = replace(base, **{k: int(best_plateau[k]) for k in keys})
    oos = run_backtest(im_os, if_os, cfg).stats
    print("\nOut-of-sample with the plateau choice:")
    for k in ("net_profit", "sharpe", "max_drawdown", "trades"):
        print(f"  {k}: {oos.get(k)}")


if __name__ == "__main__":
    main()
