# A. Intraday spread — US version of IM/IF

Same logic as the IM/IF strategy (`docs/IMIF_strategy_summary_bilingual.pdf`):
cumulative log-return spread in bp, CCI trigger, Donchian confirmation, EMA filter/exit,
RSI gate/take-profit, both legs on the same bar, notional-matched, flat before the close.
Only market specs, session times and costs change. **Every indicator threshold must be re-optimised on US data.**

## The pair

| Role | China (original) | US (this version) | Why |
|---|---|---|---|
| Leg A, "IM" | CSI 1000 futures | **Russell 2000** (RTY, micro M2K) | small caps, high beta |
| Leg B, "IF" | CSI 300 futures | **S&P 500** (ES, micro MES) | large caps, defensive |

Same economic thesis: small-vs-large style rotation swings like a pendulum, so the spread mean-reverts intraday.
Research on RTY/ES data, size with micros (M2K/MES, $5/pt each, ~$60k notional per leg).
Alpaca can't trade futures, so a paper-trading proxy would be IWM vs SPY.

Alternative pair if RTY data is missing: NQ vs ES (growth vs broad market). Weaker "pendulum" story, more trending.

## US preset (`StrategyConfig.us_rty_es()`)

- RTH only (09:30–16:00 ET); short entries before 15:00, long before 15:15, flat at 15:55
- equilibrium window 390 bars (one session); ticks RTY 0.10 / ES 0.25
- costs: 1 tick slippage per leg per fill + $0.85 per micro contract per side (check your broker)

## Run

```bash
python -m pytest
python scripts/run_backtest.py --source synthetic                       # pipeline check
python scripts/run_backtest.py --source csv --im data/RTY_1m.csv --if data/ES_1m.csv
python scripts/optimize.py --source csv --im data/RTY_1m.csv --if data/ES_1m.csv
```

CSV: `datetime, open, high, low, close`, 1-minute bars, timestamps = bar end in America/New_York,
front-month non-adjusted. `--market cn` reproduces the original IM/IF settings.

## Next

1. Convert the CME L2 data to 1-minute bars for RTY and ES (needs a sample file).
2. Re-optimise: plateau Sharpe on the first 70%, check out-of-sample on the last 30%.
3. Run long and short sides together (open item from the internship).

## Tuning run 2026-11-18 (IWM vs SPY, 2019-2026, 1-minute SIP bars)

`python scripts/fetch_alpaca_1m.py` then `python scripts/tune_us.py`. Report: [`results/2026-11-18_iwm_spy/report.md`](results/2026-11-18_iwm_spy/report.md).
Result: no parameter set with a positive edge; the IM/IF rules do not transfer to this pair as is.
