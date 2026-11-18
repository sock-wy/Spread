"""Lead-lag scan + baseline pairs backtest.

    python scripts/run_pairs.py --synthetic
    python scripts/run_pairs.py --a data/ES_mbp1.csv --b data/NQ_mbp1.csv --mult-a 50 --mult-b 20

Each CSV: timestamp column `ts` + bid_px_00, ask_px_00. Adapt once we know your format.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hfspread import leadlag, pairs, quotes  # noqa: E402
from hfspread.synthetic import synthetic_pair  # noqa: E402


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ts"]).set_index("ts").sort_index()
    return df[df["bid_px_00"] < df["ask_px_00"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a"), ap.add_argument("--b"), ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--every", default="1s")
    ap.add_argument("--mult-a", type=float, default=50.0), ap.add_argument("--mult-b", type=float, default=20.0)
    ap.add_argument("--z-in", type=float, default=2.5)
    a = ap.parse_args()

    q = synthetic_pair() if a.synthetic else quotes.align(quotes.to_mid(load(a.a), a.every), quotes.to_mid(load(a.b), a.every))
    for every in ("1s", "5s", "30s"):
        qq = q.resample(every).last().dropna()
        print(f"[{every}] {leadlag.lead_summary(leadlag.cross_corr(qq['mid_a'], qq['mid_b']))}")

    cfg = pairs.PairsConfig(mult_a=a.mult_a, mult_b=a.mult_b, z_in=a.z_in)
    trades, stats = pairs.backtest(q, cfg)
    print("\n", pd.Series(stats).round(2).to_string())
    if a.synthetic:
        print("\nSynthetic data: checks the pipeline only.")


if __name__ == "__main__":
    main()
