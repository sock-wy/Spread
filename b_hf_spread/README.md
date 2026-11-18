# B. Higher-frequency spread (baseline)

Two building blocks on 1-second top-of-book mids:

1. **Lead-lag scan** (`hfspread/leadlag.py`): cross-correlation of returns at lags ±10 samples, at 1s / 5s / 30s.
   Tells you which contract moves first and by how much.
2. **Rolling-beta pairs** (`hfspread/pairs.py`): spread = ln A − β·ln B with β from a rolling OLS,
   enter at |z| > 2.5, exit at |z| < 0.5, time stop, flat each day. Taker costs on both legs (half spread + fee).

Default pair: ES vs NQ. Swap in ES vs RTY to link with project A.

```bash
python -m pytest
python scripts/run_pairs.py --synthetic
python scripts/run_pairs.py --a data/ES.csv --b data/NQ.csv --mult-a 50 --mult-b 20
```

Next steps once the baseline numbers are in:
- add the OFI / imbalance filter from project C (enter only when the book agrees)
- passive entries on the slower leg (lead-lag: buy the laggard with a limit order)
- Kalman-filter β; calendar spread (front vs next month) during roll weeks
- realistic fills via [hftbacktest](https://github.com/nkaz001/hftbacktest) (queue + latency)
