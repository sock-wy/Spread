"""Run the baseline order-book study.

    python scripts/run_study.py --synthetic
    python scripts/run_study.py --csv data/ES_mbp10.csv --tick 0.25

CSV needs a timestamp column `ts` plus bid_px_00, bid_sz_00, ask_px_00, ask_sz_00
(more levels optional). Adapt `load()` once we know your file format.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lob import features, study  # noqa: E402
from lob.synthetic import synthetic_book  # noqa: E402


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ts"]).set_index("ts").sort_index()
    return df[(df["bid_px_00"] < df["ask_px_00"])]          # drop crossed / empty books


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"), ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--tick", type=float, default=0.25)
    ap.add_argument("--every", default="1s")
    ap.add_argument("--horizons", default="1,5,10,30")
    a = ap.parse_args()

    raw = synthetic_book() if a.synthetic else load(a.csv)
    feat = features.all_features(raw)
    sampled = study.sample(feat, a.every)
    cols = [c for c in ("imbalance_l1", "imbalance_d5", "ofi") if c in sampled]
    sampled["micro_minus_mid"] = (sampled["microprice"] - sampled["mid"]) / a.tick
    cols.append("micro_minus_mid")
    print(f"{len(raw):,} updates -> {len(sampled):,} samples at {a.every}; "
          f"median spread {sampled['spread'].median() / a.tick:.2f} ticks\n")
    for h in map(int, a.horizons.split(",")):
        print(f"--- horizon {h} x {a.every} ---")
        print(study.evaluate(study.add_target(sampled, h, a.tick), cols, a.tick).round(4).to_string(), "\n")


if __name__ == "__main__":
    main()
