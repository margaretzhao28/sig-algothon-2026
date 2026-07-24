"""
Robustness and overfitting-detection tools.

All functions return a list of dicts so results can be aggregated, printed,
or passed directly to matplotlib without coupling to the display layer.
"""

from __future__ import annotations
import warnings
from typing import Callable, Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from .engine import (
    run_backtest, BacktestResult, default_eval_window,
    COMM_RATE, DLR_POS_LIMIT, RESEARCH_FRAC,
)
from .metrics import compute_metrics, official_score
from .data import walk_forward_windows


# ---------------------------------------------------------------------------
# Walk-forward evaluation
# ---------------------------------------------------------------------------

def walk_forward(
    prices: np.ndarray,
    strategy: Callable[[np.ndarray], np.ndarray],
    *,
    train_size: int = 500,
    test_size: int = 125,
    step: int | None = None,
    min_train: int | None = None,
    comm_rate: float = COMM_RATE,
    dlr_pos_limit: float = DLR_POS_LIMIT,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Evaluate strategy on successive out-of-sample test windows.

    IMPORTANT — how "training" works here.  This framework mirrors the live competition
    contract: getMyPosition(prices[:, :t]) is always called with the FULL price history
    up to day t.  There is no separate fit step the framework can restrict.  Consequently
    `train_size` only controls WHERE each test window starts (and is reported as
    `train_range` for context); it does NOT limit what history the strategy sees on a
    test day.  Two runs with different `train_size` but the same test window produce
    identical scores.  If your strategy should only look back N days, enforce that inside
    getMyPosition (e.g. prices[:, -N:]) — the engine will not do it for you.

    What this DOES give you: an honest out-of-sample read of the strategy across multiple,
    non-overlapping market periods — the best available proxy for the unseen eval set.

    Returns a list of dicts, one per fold, each containing:
      fold, train_range, test_range, score, mean_pl, std_pl, ann_sharpe,
      total_dvolume, hit_rate, metrics (full metrics dict)
    """
    nInst, nt = prices.shape
    results = []

    windows = list(walk_forward_windows(
        nt, train_size=train_size, test_size=test_size,
        step=step, min_train=min_train,
    ))

    for fold_idx, (tr_start, tr_end, te_start, te_end) in enumerate(windows):
        r = run_backtest(
            prices,
            strategy,
            eval_start=te_start + 1,   # +1 because engine passes prices[:, :t]
            eval_end=te_end + 1,
            comm_rate=comm_rate,
            dlr_pos_limit=dlr_pos_limit,
            verbose=False,
        )
        m = compute_metrics(r)
        fold_result = {
            "fold": fold_idx,
            "train_range": (tr_start, tr_end),
            "test_range": (te_start, te_end),
            "score": m["score"],
            "mean_pl": m["mean_pl"],
            "std_pl": m["std_pl"],
            "ann_sharpe": m["ann_sharpe"],
            "total_dvolume": m["total_dvolume"],
            "hit_rate": m["hit_rate"],
            "metrics": m,
            "result": r,
        }
        if verbose:
            print(
                f"Fold {fold_idx:2d}  train=[{tr_start},{tr_end})  "
                f"test=[{te_start},{te_end})  "
                f"score={m['score']:8.2f}  sharpe={m['ann_sharpe']:6.2f}"
            )
        results.append(fold_result)

    return results


def walk_forward_summary(folds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate walk-forward fold results into a summary dict."""
    scores = np.array([f["score"] for f in folds])
    sharpes = np.array([f["ann_sharpe"] for f in folds])
    return {
        "n_folds": len(folds),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores, ddof=0)),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_positive_frac": float(np.mean(scores > 0)),
        "sharpe_mean": float(np.mean(sharpes)),
        "scores": scores,
        "sharpes": sharpes,
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence interval on score
# ---------------------------------------------------------------------------

def _stationary_bootstrap_indices(
    n: int, expected_block: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Politis & Romano (1994) stationary bootstrap index sequence of length n.

    Blocks have geometric length (mean = expected_block) and wrap around the series,
    preserving short-range serial dependence — essential for time-series PL where
    i.i.d. resampling underestimates variance and yields over-narrow CIs.
    """
    p = 1.0 / max(expected_block, 1.0)        # restart probability
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(0, n)
    for t in range(1, n):
        if rng.random() < p:
            idx[t] = rng.integers(0, n)        # start a new block
        else:
            idx[t] = (idx[t - 1] + 1) % n      # continue current block (wrap)
    return idx


def bootstrap_score_ci(
    result: BacktestResult,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    method: str = "stationary",
    block_size: Optional[float] = None,
    seed: int | None = 42,
) -> Dict[str, Any]:
    """
    Non-parametric bootstrap confidence interval for the competition score.

    Parameters
    ----------
    method     : "stationary" (default) uses a block bootstrap that preserves the
                 autocorrelation of daily PL — the correct choice for a time series.
                 "iid" resamples days independently; it IGNORES serial correlation and
                 produces a deceptively tight CI.  Use "iid" only as a contrast.
    block_size : expected block length for the stationary bootstrap.
                 Defaults to n**(1/3) (a standard rule of thumb).

    Returns point estimate, CI bounds, and the full bootstrap distribution.
    """
    rng = np.random.default_rng(seed)
    pl = result.daily_pl
    n = len(pl)

    if block_size is None:
        block_size = max(2.0, round(n ** (1.0 / 3.0)))

    boot_scores = np.empty(n_bootstrap)
    if method == "iid":
        for i in range(n_bootstrap):
            boot_scores[i] = official_score(rng.choice(pl, size=n, replace=True))
    elif method == "stationary":
        for i in range(n_bootstrap):
            idx = _stationary_bootstrap_indices(n, block_size, rng)
            boot_scores[i] = official_score(pl[idx])
    else:
        raise ValueError(f"Unknown bootstrap method: {method!r} (use 'stationary' or 'iid')")

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_scores, 100 * alpha / 2))
    upper = float(np.percentile(boot_scores, 100 * (1 - alpha / 2)))
    point = official_score(pl)
    return {
        "score": point,
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence": confidence,
        "method": method,
        "block_size": float(block_size) if method == "stationary" else None,
        "prob_positive": float(np.mean(boot_scores > 0)),
        "n_bootstrap": n_bootstrap,
        "boot_scores": boot_scores,   # full distribution for plotting
    }


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

def parameter_sweep(
    prices: np.ndarray,
    strategy_factory: Callable[..., Callable[[np.ndarray], np.ndarray]],
    param_grid: Dict[str, Sequence[Any]],
    *,
    is_window: Optional[Tuple[int, int]] = None,
    oos_window: Optional[Tuple[int, int]] = None,
    research_frac: float = RESEARCH_FRAC,
    comm_rate: float = COMM_RATE,
    dlr_pos_limit: float = DLR_POS_LIMIT,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Grid-search over strategy parameters with built-in overfitting controls.

    *** SELECTION BIAS WARNING ***
    Tuning on a single window and reporting the best score is the #1 way to produce a
    backtest that looks great and then fails live.  The more combinations you try, the
    higher the best in-sample score climbs by pure chance.  This function therefore:
      1. Selects parameters on an IN-SAMPLE window (`is_window`).
      2. Re-scores every combination on a held-out OUT-OF-SAMPLE window (`oos_window`).
      3. Reports the IS->OOS degradation so you can see how much of the edge was luck.
    Trust the OOS column, not the IS column.  If OOS collapses, the params are overfit.

    Parameters
    ----------
    strategy_factory : callable(**params) -> getMyPosition function
    param_grid       : dict mapping param name -> list of values to try
    is_window        : (start, end) eval days for selection. Default: the second
                       quarter-to-half of the research period.
    oos_window       : (start, end) out-of-sample eval days. Default: the unseen
                       remainder after research_frac (mirrors run_backtest's default).

    Returns
    -------
    dict with:
      results          : list of per-combo dicts (is_score, oos_score, ...), IS-sorted
      best_is          : the IS-best combo (what naive tuning would pick)
      best_oos         : the OOS-best combo (the honest winner)
      n_trials         : number of combinations evaluated
      is_oos_corr      : rank/linear correlation between IS and OOS scores (low = overfit)
      expected_max_is_inflation : rough estimate of how much the IS-best is inflated by
                                  multiple testing (std of IS scores * sqrt(2 ln n_trials))
    """
    from itertools import product

    nt = prices.shape[1]
    split, _end = default_eval_window(nt, research_frac)
    if oos_window is None:
        oos_window = (split, nt)
    if is_window is None:
        # In-sample selection window: the second quarter-to-half of the training period,
        # leaving the first part as warmup history.  Falls back gracefully for small nt.
        is_start = max(2, split // 2)
        is_window = (is_start, split)

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combos = list(product(*values))
    n_trials = len(combos)

    if n_trials > 1:
        warnings.warn(
            f"parameter_sweep is evaluating {n_trials} combinations on the same data. "
            "The best IN-SAMPLE score is upward-biased by multiple testing; rely on the "
            "OUT-OF-SAMPLE column and walk-forward validation before trusting any pick.",
            stacklevel=2,
        )

    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        strat = strategy_factory(**params)

        r_is = run_backtest(prices, strat, eval_start=is_window[0], eval_end=is_window[1],
                            comm_rate=comm_rate, dlr_pos_limit=dlr_pos_limit)
        r_oos = run_backtest(prices, strat, eval_start=oos_window[0], eval_end=oos_window[1],
                             comm_rate=comm_rate, dlr_pos_limit=dlr_pos_limit)
        m_is = compute_metrics(r_is)
        m_oos = compute_metrics(r_oos)
        entry = {
            "params": params,
            "is_score": m_is["score"],
            "oos_score": m_oos["score"],
            "is_sharpe": m_is["ann_sharpe"],
            "oos_sharpe": m_oos["ann_sharpe"],
            "degradation": m_is["score"] - m_oos["score"],
        }
        if verbose:
            pstr = ", ".join(f"{k}={v}" for k, v in params.items())
            print(f"  {pstr:<40}  IS={m_is['score']:8.3f}  OOS={m_oos['score']:8.3f}")
        results.append(entry)

    results.sort(key=lambda x: x["is_score"], reverse=True)
    is_scores = np.array([e["is_score"] for e in results])
    oos_scores = np.array([e["oos_score"] for e in results])

    is_oos_corr = (
        float(np.corrcoef(is_scores, oos_scores)[0, 1])
        if n_trials > 1 and is_scores.std() > 0 and oos_scores.std() > 0 else float("nan")
    )
    expected_inflation = (
        float(is_scores.std(ddof=0) * np.sqrt(2.0 * np.log(n_trials)))
        if n_trials > 1 else 0.0
    )

    return {
        "results": results,
        "best_is": results[0],
        "best_oos": max(results, key=lambda x: x["oos_score"]),
        "n_trials": n_trials,
        "is_window": is_window,
        "oos_window": oos_window,
        "is_oos_corr": is_oos_corr,
        "expected_max_is_inflation": expected_inflation,
    }
