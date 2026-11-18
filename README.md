# Spread

Three trading-research projects plus shared paper-trading plumbing.

| Folder | Project | Status |
|---|---|---|
| `a_intraday_spread/` | **A.** Intraday small-cap vs large-cap futures spread (US version of the IM/IF strategy) | Code ported, US preset added; needs data + re-optimisation |
| `b_hf_spread/` | **B.** Higher-frequency spread: lead-lag scan + rolling-beta pairs on 1s mids | Baseline skeleton, runs on synthetic data |
| `c_orderbook/` | **C.** Order book: L2 book, imbalance / microprice / OFI, short-horizon prediction study | Baseline skeleton, runs on synthetic data |
| `common/` | Alpaca paper-account helpers (credentials, bars, risk checks, order submit, CSV log) | Working |

## Setup

```bash
pip install -r requirements.txt
export APCA_API_KEY_ID=...        # paper keys, kept in ~/.zshrc, never in code
export APCA_API_SECRET_KEY=...
```

Run each project's tests from its folder: `cd a_intraday_spread && python -m pytest`.

## Data

CME L2 data (about six months) is the main input for all three. Raw data is not committed (`data/` is git-ignored).
Each project's loader expects Databento-style column names (`bid_px_00`, `bid_sz_00`, `ask_px_00`, ...);
a converter for the actual file format is the next step.
