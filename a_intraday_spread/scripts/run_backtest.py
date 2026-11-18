"""Run the IM/IF spread backtest.

Examples
--------
Synthetic demo data (no data licence needed):
    python scripts/run_backtest.py --source synthetic

Your own minute bars:
    python scripts/run_backtest.py --source csv --im data/IM88.csv --if data/IF88.csv

RiceQuant (requires rqdatac and a licence):
    python scripts/run_backtest.py --source rqdata --start 2022-07-22 --end 2024-07-22
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imif import StrategyConfig, run_backtest  # noqa: E402
from imif.data import align_legs, filter_rth, load_csv, load_rqdata, synthetic_legs, validate_bars  # noqa: E402
from imif.metrics import check_targets  # noqa: E402
from imif.plotting import plot_result  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "csv", "rqdata"], default="synthetic")
    ap.add_argument("--im"), ap.add_argument("--if", dest="if_")
    ap.add_argument("--start"), ap.add_argument("--end")
    ap.add_argument("--days", type=int, default=120, help="synthetic only")
    ap.add_argument("--out", default="output")
    ap.add_argument("--short-only", action="store_true")
    ap.add_argument("--market", choices=["cn", "us"], default="us",
                    help="us = M2K/MES preset (RTH, US costs); cn = original IM/IF settings")
    args = ap.parse_args()

    if args.source == "synthetic":
        im, if_ = (synthetic_legs(n_days=args.days, session="us", im0=2400, if0=6000, tick=0.25)
                   if args.market == "us" else synthetic_legs(n_days=args.days))
    elif args.source == "csv":
        im, if_ = load_csv(args.im, args.if_)
        if args.market == "us":
            im, if_ = filter_rth(im), filter_rth(if_)
    else:
        im, if_ = load_rqdata(args.start, args.end)
    im, if_ = align_legs(im, if_)
    for issue in validate_bars(im, "IM") + validate_bars(if_, "IF"):
        print("WARNING:", issue)

    cfg = StrategyConfig.us_rty_es() if args.market == "us" else StrategyConfig()
    cfg.enable_long = not args.short_only
    res = run_backtest(im, if_, cfg)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.trades.to_csv(out / "trades.csv", index=False)
    res.equity.to_csv(out / "equity.csv")
    (out / "stats.json").write_text(json.dumps({"source": args.source, "config": cfg.to_dict(), "stats": res.stats}, indent=2, default=str))
    plot_result(res, out / "backtest.png", title=f"IM/IF spread backtest ({args.source} data)")

    print(json.dumps(res.stats, indent=2, default=str))
    print(check_targets(res.stats).to_string(index=False))
    if args.source == "synthetic":
        print("\nNote: synthetic data checks that the pipeline runs end to end. Its P&L says nothing about the strategy.")


if __name__ == "__main__":
    main()
