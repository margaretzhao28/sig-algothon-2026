# Pairs research log

## Data discipline

- **Days 1–500:** discovery and pair screening.
- **Days 501–700:** validation and parameter selection.
- **Days 701–1000:** untouched final test. Do not use these results to revisit pair selection or tuning.

## Current pair-selection process

- Exclude ALGO from all pairs.
- Screen both pair orientations using return correlation, Engle–Granger p-value, ADF p-value of the spread, half-life, zero-crossings, and beta stability.
- Resolve ticker conflicts globally: a stock can appear in only one selected pair.
- This produced 12 validated, non-overlapping pairs. They were ranked by validation leave-one-out contribution and compared at portfolio sizes 3, 5, 8, 9, 10, 11, and 12; the top-11 portfolio is the current execution-research book.

## Current execution findings

| Configuration | Validation score | Decision note |
| --- | ---: | --- |
| Top-11 equal dollar, daily beta refit | 399.45 | Current score baseline. |
| Top-11 equal dollar, 5-day beta refit | 392.37 | Lower than baseline. |
| Top-11 equal dollar, 10-day beta refit | 398.17 | Almost identical score, with slightly higher Sharpe and smaller drawdown; worth final testing. |
| Top-11 rolling beta hedge, daily refit | 349.28 | Lower score; equal-dollar sizing is the current preferred baseline. |

## Next final-test runs

Run the daily-, 5-day-, and 10-day-refit equal-dollar strategies once on days 701–1000. Record each result in `experiments/experiment_log.csv`, then choose a deployment candidate without retuning from the test outcomes.
