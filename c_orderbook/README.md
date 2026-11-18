# C. Order book (baseline)

1. **Book** (`lob/book.py`): price-level L2 book from incremental updates (CME MBP / Databento `mbp-*`).
2. **Features** (`lob/features.py`): mid, spread, L1 and depth imbalance, microprice, OFI (Cont, Kukanov & Stoikov 2014).
3. **Study** (`lob/study.py`): sample at 1s, predict the mid move over the next 1/5/10/30s,
   report out-of-sample R², sign hit rate, and how often the predicted move beats half the spread.

```bash
python -m pytest
python scripts/run_study.py --synthetic
python scripts/run_study.py --csv data/ES_mbp10.csv --tick 0.25
```

The study answers one question first: is there a short-horizon signal in this book, and is it larger than the cost of trading on it?

Next steps:
- imbalance-driven taker strategy (trade only when prediction > half spread + fee)
- queue-position simulation for passive orders (L2 only: estimate the queue; links to FIN 556 / IE 421 plans)
- C++ rewrite of `book.py` with latency measurement once the Python logic is validated
