# Experiment log

Used `experiment_log.csv` to record each meaningful backtest across pairs, mean-reversion, momentum, and future strategy research. Each row captures the strategy version, universe, parameters, evaluation period, core metrics, and a short note on what changed or what I learned.

Log is kept as record of research decisions. Generated backtest outputs can live in the ignored `results/` directory, with their location recorded in `result_path` when relevant.

## Pairs research workflow

1. Used days 1–500 for pair discovery and screening, producing **12 validated, non-overlapping pairs**.
2. Used days 501–700 for validation. The 12-pair portfolio was ranked using leave-one-out contribution and tested at portfolio sizes **3, 5, 8, 9, 10, 11, and 12**.
3. Compared beta-hedged and equal-dollar sizing using the top-11 pair portfolio. **Equal-dollar sizing** was the stronger validation baseline.
4. Tested execution controls on the fixed top-11 equal-dollar portfolio. Daily beta refitting achieved the highest validation score, while a **10-day refit** was very close and had slightly better Sharpe and drawdown.

Days 701–1000 are reserved for final testing. These evaluations will be recorded here but will not be used to revisit pair selection or tune parameters.

## Pairs strategy naming

- `pairs_top_*_rolling_beta.py` (including `pairs_top_10_rolling_beta.py`) uses **beta-hedged sizing**: beta determines both the spread signal and relative leg sizes.
- `pairs_top_*_equal_dollar_*.py` uses **equal-dollar sizing**: beta determines the spread signal only, while each leg targets approximately $10k.

## Log fields

| Strategy type | `assets_or_universe` example | `parameters` example |
| --- | --- | --- |
| Pairs | `GARI-EELT; SMAH-ILVX; ...` | `lookback=250; entry_z=1.5; exit_z=0.5; max_hold=30` |
| Mean reversion | `MMBT, EELT` | `window=20; entry_z=1.5; exit_z=0.5; max_hold=10` |
| Momentum | `all 50 stocks` | `lookback=60; rebalance=5; long_short=5` |
