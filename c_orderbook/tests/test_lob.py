import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lob import features, study  # noqa: E402
from lob.book import L2Book  # noqa: E402
from lob.synthetic import synthetic_book  # noqa: E402


def test_book_updates_and_snapshot():
    b = L2Book(tick=0.25)
    b.update("B", 6000.00, 10); b.update("B", 5999.75, 5)
    b.update("A", 6000.25, 7); b.update("A", 6000.50, 3)
    assert b.best_bid() == (6000.0, 10) and b.best_ask() == (6000.25, 7)
    b.update("B", 6000.00, 0)                       # level removed
    assert b.best_bid() == (5999.75, 5) and not b.is_crossed()
    snap = b.snapshot(2)
    assert snap["ask_px_01"] == 6000.5 and snap["bid_sz_01"] == 0.0


def test_ofi_sign():
    df = pd.DataFrame({"bid_px_00": [100, 100, 100.25], "bid_sz_00": [10, 15, 5],
                       "ask_px_00": [100.25, 100.25, 100.5], "ask_sz_00": [10, 10, 8]})
    e = features.ofi(df)
    assert e.iloc[1] == 5          # bid size grew at same price -> buying pressure
    assert e.iloc[2] > 0           # bid stepped up, ask stepped up -> buying pressure


def test_microprice_leans_to_thin_side():
    df = pd.DataFrame({"bid_px_00": [100.0], "bid_sz_00": [90.0], "ask_px_00": [100.25], "ask_sz_00": [10.0]})
    f = features.basic(df)
    assert f["microprice"].iloc[0] > f["mid"].iloc[0]


def test_study_finds_signal_in_synthetic_book():
    raw = synthetic_book()
    s = study.add_target(study.sample(features.all_features(raw)), horizon=5, tick=0.25)
    res = study.evaluate(s, ["imbalance_l1", "ofi"], tick=0.25)
    assert res.loc["imbalance_l1", "hit_rate"] > 0.55
    assert np.isfinite(res["oos_r2"]).all()
