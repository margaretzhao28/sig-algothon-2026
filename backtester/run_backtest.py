#!/usr/bin/env python
"""
Command-line entry point for the backtest framework.

Usage examples
--------------
# Full run — everything saved to ./results/
    python run_backtest.py --full

# Main eval only (default: research on first 50%, evaluate the unseen rest):
    python run_backtest.py

# Test on a specific sub-window, leaving the remainder untouched:
    python run_backtest.py --eval-start 500 --eval-end 750

# Official 2026 eval.py window (last 250 days, e.g. 250-500 on 500-day data):
    python run_backtest.py --eval-start 250 --eval-end 500

# Walk-forward CV with stats saved:
    python run_backtest.py --walk-forward --out-dir results/

# Bootstrap CI:
    python run_backtest.py --bootstrap --n-bootstrap 2000

# Save plots:
    python run_backtest.py --plots report.pdf
"""

import argparse
import importlib
import importlib.util
import json
import os
import pickle
import sys
from datetime import datetime

# Re-exec under the project venv so `python run_backtest.py ...` works from any
# shell without `source .venv/bin/activate` — the venv has numpy/PyQt6/etc.
_VENV_PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            ".venv", "bin", "python")
if (os.path.exists(_VENV_PYTHON)
        and os.path.realpath(sys.executable) != os.path.realpath(_VENV_PYTHON)):
    os.execv(_VENV_PYTHON, [_VENV_PYTHON] + sys.argv)

import numpy as np
import pandas as pd

from backtest import (
    load_prices, run_backtest, compute_metrics,
    print_report, save_plots,
    walk_forward, bootstrap_score_ci,
    prices_path,
)
from backtest.engine import RESEARCH_FRAC
from backtest.report import print_walk_forward_summary
from backtest.robustness import walk_forward_summary


# ---------------------------------------------------------------------------
# Saving stats to disk
# ---------------------------------------------------------------------------

def _serialisable(obj):
    """Recursively make a metrics dict JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialisable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) or np.isinf(v) else v
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def save_stats(
    out_dir: str,
    metrics: dict,
    wf_folds: list | None,
    boot_ci: dict | None,
    result,
    run_meta: dict,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. Main metrics JSON ---
    # Strip large arrays that belong in plots, not JSON
    _ARRAY_KEYS = {"rolling_vol_20d", "rolling_sharpe_20d", "inst_pl_total",
                   "inst_sharpe", "turnover_ratio", "boot_scores"}
    scalar_metrics = {k: v for k, v in metrics.items() if k not in _ARRAY_KEYS}
    payload = {"run": run_meta, "metrics": _serialisable(scalar_metrics)}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(payload, f, indent=2)

    # --- 2. Per-instrument stats CSV ---
    inst_df = pd.DataFrame({
        "instrument": range(result.nInst),
        "total_pl":   metrics["inst_pl_total"].tolist(),
        "ann_sharpe": metrics["inst_sharpe"].tolist(),
    }).sort_values("total_pl", ascending=False)
    inst_df.to_csv(os.path.join(out_dir, "instrument_stats.csv"), index=False)

    # --- 3. Daily PL series CSV ---
    daily_df = pd.DataFrame({
        "day":         result.eval_days,
        "pl":          result.daily_pl,
        "cum_value":   result.cumulative_value,
        "dvolume":     result.daily_dvolume,
        "commission":  result.daily_commission,
    })
    daily_df.to_csv(os.path.join(out_dir, "daily_pl.csv"), index=False)

    # --- 4. Walk-forward fold table CSV ---
    if wf_folds:
        rows = []
        for f in wf_folds:
            m = f["metrics"]
            rows.append({
                "fold":                    f["fold"],
                "test_start":              f["test_range"][0],
                "test_end":                f["test_range"][1],
                "score":                   f["score"],
                "mean_pl":                 f["mean_pl"],
                "std_pl":                  f["std_pl"],
                "ann_sharpe":              f["ann_sharpe"],
                "sortino":                 m["sortino"],
                "calmar_ratio":            m["calmar_ratio"],
                "omega_ratio":             m["omega_ratio"],
                "profit_factor":           m["profit_factor"],
                "hit_rate":                m["hit_rate"],
                "var_95":                  m["var_95"],
                "cvar_95":                 m["cvar_95"],
                "max_drawdown":            m["max_drawdown"],
                "max_drawdown_duration":   m["max_drawdown_duration_days"],
                "pct_time_underwater":     m["pct_time_underwater"],
                "ulcer_index":             m["ulcer_index"],
                "skewness":                m["skewness"],
                "kurtosis_excess":         m["kurtosis_excess"],
                "autocorr_lag1":           m["autocorr_lag1"],
                "max_consec_loss_days":    m["max_consec_loss_days"],
                "avg_daily_turnover":      m["avg_daily_turnover_ratio"],
                "total_dvolume":           m["total_dvolume"],
            })
        pd.DataFrame(rows).to_csv(
            os.path.join(out_dir, "walk_forward_folds.csv"), index=False
        )

        wf_summary = walk_forward_summary(wf_folds)
        with open(os.path.join(out_dir, "walk_forward_summary.json"), "w") as f:
            json.dump(_serialisable({
                "run": run_meta,
                "summary": {k: v for k, v in wf_summary.items()
                            if k not in ("scores", "sharpes")},
                "score_by_fold": wf_summary["scores"].tolist(),
                "sharpe_by_fold": wf_summary["sharpes"].tolist(),
            }), f, indent=2)

    # --- 5. Bootstrap CI JSON ---
    if boot_ci:
        with open(os.path.join(out_dir, "bootstrap_ci.json"), "w") as f:
            json.dump(_serialisable({
                "run": run_meta,
                "score": boot_ci["score"],
                "ci_lower": boot_ci["ci_lower"],
                "ci_upper": boot_ci["ci_upper"],
                "confidence": boot_ci["confidence"],
                "method": boot_ci["method"],
                "block_size": boot_ci["block_size"],
                "prob_positive": boot_ci["prob_positive"],
                "n_bootstrap": boot_ci["n_bootstrap"],
            }), f, indent=2)

    print(f"\n[stats] Saved to '{out_dir}/':")
    for fname in sorted(os.listdir(out_dir)):
        size = os.path.getsize(os.path.join(out_dir, fname))
        print(f"  {fname:<35}  {size:>7,} bytes")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Algothon backtest framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--prices",    default=None,
                   help="Path to prices data file (overrides --dataset)")
    p.add_argument("--dataset",   default="2026",
                   help="Shortcut for --prices data/<dataset>/prices.txt")
    p.add_argument("--strategy",  default="main",       help="Module containing getMyPosition")
    p.add_argument("--max-days",  type=int, default=None,
                   help="Truncate price history to first N days (treats later data as unseen)")

    # Eval window — by default, score everything after the research period (the
    # unseen "rest"). Pass --eval-start/--eval-end to score an explicit sub-window
    # instead (e.g. to test on part of the unseen data and deliberately leave the
    # remainder untouched — that discipline is on you; the framework doesn't enforce it).
    p.add_argument("--research-frac", type=float, default=None,
                   help=f"Fraction of days reserved for research/design; default eval "
                        f"window is the unseen rest (default {RESEARCH_FRAC})")
    p.add_argument("--eval-start", type=int, default=None,
                   help="Explicit first day t to score (overrides --research-frac; for eval.py parity)")
    p.add_argument("--eval-end",   type=int, default=None,
                   help="Explicit exclusive upper bound (overrides --research-frac)")
    # Market params
    p.add_argument("--comm-rate", type=float, default=0.0001,
                   help="Commission rate (0.0001 = 1bp; instrument 0 auto-overridden to 0.2bp)")
    p.add_argument("--pos-limit", type=float, default=10000.0,
                   help="Dollar position limit per instrument (instrument 0 auto-overridden to $100k)")

    # Robustness
    p.add_argument("--walk-forward", action="store_true",
                   help="Walk-forward out-of-sample evaluation")
    p.add_argument("--train-size", type=int, default=500,
                   help="Walk-forward: training window (days)")
    p.add_argument("--test-size",  type=int, default=125,
                   help="Walk-forward: test window (days)")
    p.add_argument("--step",       type=int, default=None,
                   help="Walk-forward: step size (default=test-size)")

    p.add_argument("--bootstrap",   action="store_true", help="Bootstrap CI on score")
    p.add_argument("--n-bootstrap", type=int, default=2000, help="Bootstrap iterations")
    p.add_argument("--boot-method", choices=["stationary", "iid"], default="stationary",
                   help="Bootstrap method: 'stationary' preserves PL autocorrelation")

    # Output
    p.add_argument("--out-dir", type=str, default=None,
                   help="Directory to save stats (JSON/CSV). Auto-named if --full.")
    p.add_argument("--plots",   type=str, default=None,
                   help="Save plots to this path (.pdf or .png)")
    p.add_argument("--no-viz", action="store_true",
                   help="Skip the interactive HTML visualizer (auto-opens by default)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Print per-day output")

    # Shortcut
    p.add_argument("--full", action="store_true",
                   help="Full run: walk-forward + bootstrap + plots + save all stats")

    return p.parse_args()


def _launch_dashboard() -> None:
    """Open the PyQt6 dashboard on the run we just saved (results/last_run.pkl)."""
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    app = os.path.join(here, "viz_qt.py")
    # Prefer the project venv's python: PyQt6 lives there, and sys.executable may
    # be a different interpreter (system/conda) that can run the backtest but not
    # the dashboard.
    venv_python = os.path.join(here, ".venv", "bin", "python")
    python = venv_python if os.path.exists(venv_python) else sys.executable
    log_path = os.path.join(here, "results", "viz.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as log:
        subprocess.Popen(
            [python, app],
            stdout=log, stderr=log,
            start_new_session=True,
        )
    print("\nOpening dashboard window... (already open? hit its ⟳ Reload button; "
          "--no-viz to skip; errors go to results/viz.log)")


def main() -> None:
    args = parse_args()

    # --dataset is a shortcut that resolves to data/<dataset>/prices.txt;
    # an explicit --prices path wins over it.
    if args.prices is None:
        args.prices = prices_path(args.dataset)

    # Namespace output by dataset (inferred from the resolved path's parent dir,
    # e.g. data/2026/prices.txt -> "2026") so runs on different datasets never collide.
    dataset_label = os.path.basename(
        os.path.dirname(os.path.realpath(args.prices))
    )

    # --full expands to everything with auto output directory
    if args.full:
        args.walk_forward = True
        args.bootstrap    = True
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.out_dir is None:
            args.out_dir = os.path.join("results", dataset_label, ts)
        if args.plots is None:
            args.plots = os.path.join(args.out_dir, "report.pdf")

    # Load prices
    print(f"Loading prices from '{args.prices}' ...")
    prices = load_prices(args.prices)
    nInst, nt = prices.shape
    if args.max_days is not None and args.max_days < nt:
        prices = prices[:, : args.max_days]
        nt = args.max_days
        print(f"Loaded {nInst} instruments x {nt} days  (truncated to first {nt} days)\n")
    else:
        print(f"Loaded {nInst} instruments x {nt} days\n")

    # Load strategy — accepts a module name (e.g. "anso_mm") or a file path
    # (e.g. "../Sean/strategies/ANSO-mm.py").
    print(f"Loading strategy '{args.strategy}' ...")
    try:
        if args.strategy.endswith(".py") or os.sep in args.strategy:
            path = os.path.abspath(args.strategy)
            mod_name = os.path.splitext(os.path.basename(path))[0].replace("-", "_")
            spec = importlib.util.spec_from_file_location(mod_name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            args.strategy = mod_name  # used in run metadata / dashboard title
        else:
            mod = importlib.import_module(args.strategy)
        strategy = mod.getMyPosition
        fair_price_fn = getattr(mod, "getFairPrice", None)
        print("Strategy loaded."
              + (" (with getFairPrice for visualization)" if fair_price_fn else "")
              + "\n")
    except (ImportError, AttributeError, FileNotFoundError) as e:
        print(f"ERROR: Could not load strategy: {e}")
        sys.exit(1)

    # Resolve the evaluation window.
    # Default: evaluate everything after the research period (--research-frac sets its
    # size). --eval-start / --eval-end override explicitly — e.g. to test on a specific
    # sub-window and deliberately leave the remainder untouched (your responsibility;
    # the framework doesn't enforce a seal), or to reproduce eval.py's exact range.
    research_frac = args.research_frac if args.research_frac is not None else RESEARCH_FRAC

    if args.eval_start is not None or args.eval_end is not None:
        _es = args.eval_start if args.eval_start is not None else 0
        _ee = args.eval_end if args.eval_end is not None else nt
    else:
        _es = max(1, min(int(round(nt * research_frac)), nt - 1))
        _ee = nt

    run_meta = {
        "timestamp":     datetime.now().isoformat(),
        "strategy":      args.strategy,
        "dataset":       dataset_label,
        "prices_file":   args.prices,
        "nInst":         nInst,
        "nt":            nt,
        "research_frac": research_frac,
        "eval_start":    _es,
        "eval_end":      _ee,
        "n_eval":        _ee - _es,
        "comm_rate":     args.comm_rate,
        "pos_limit":     args.pos_limit,
    }

    print(
        f"Evaluating on days {_es}–{_ee - 1}  "
        f"({_ee - _es} days; researched on first {_es} of {nt})"
    )
    result  = run_backtest(
        prices, strategy,
        eval_start=_es, eval_end=_ee,
        comm_rate=args.comm_rate,
        dlr_pos_limit=args.pos_limit,
        verbose=args.verbose,
        fair_price_fn=fair_price_fn,
    )
    metrics = compute_metrics(result)
    print_report(result, metrics)

    # Ridge-tail rank note for the hover (only needed when the viz will open).
    pos_annot = None
    if not args.no_viz:
        try:
            from backtest.visualize import ridge_rank_annotations
            pos_annot = ridge_rank_annotations(prices, result.eval_days, mod)
        except Exception as exc:                       # never block a run on the hover
            print(f"  (ridge-rank hover skipped: {exc})")

    # Persist this run for the dashboard, then open/refresh it
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "last_run.pkl"), "wb") as f:
        pickle.dump({
            "result": result,
            "metrics": metrics,
            "strategy": args.strategy,
            "dataset": dataset_label,
            "prices_file": args.prices,
            "pos_annot": pos_annot,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }, f)

    if not args.no_viz:
        _launch_dashboard()

    wf_folds = None
    boot_ci  = None

    if args.walk_forward:
        print(f"\nRunning walk-forward CV (train={args.train_size}, test={args.test_size}) ...")
        wf_folds = walk_forward(
            prices, strategy,
            train_size=args.train_size, test_size=args.test_size, step=args.step,
            comm_rate=args.comm_rate, dlr_pos_limit=args.pos_limit,
            verbose=True,
        )
        print_walk_forward_summary(wf_folds)

    if args.bootstrap:
        print(f"\nComputing bootstrap score CI (n={args.n_bootstrap}, method={args.boot_method}) ...")
        boot_ci = bootstrap_score_ci(
            result, n_bootstrap=args.n_bootstrap, method=args.boot_method,
        )
        bs = f", block~{boot_ci['block_size']:.0f}d" if boot_ci["block_size"] else ""
        print(f"  Score        : {boot_ci['score']:.4f}")
        print(f"  95% CI       : [{boot_ci['ci_lower']:.4f}, {boot_ci['ci_upper']:.4f}]  "
              f"({boot_ci['method']}{bs})")
        print(f"  P(score > 0) : {boot_ci['prob_positive']*100:.1f}%")

    if args.plots:
        os.makedirs(os.path.dirname(args.plots) or ".", exist_ok=True)
        print(f"\nSaving plots to '{args.plots}' ...")
        save_plots(result, metrics, output_path=args.plots,
                   wf_folds=wf_folds, boot_ci=boot_ci)

    if args.out_dir:
        save_stats(args.out_dir, metrics, wf_folds, boot_ci, result, run_meta)


if __name__ == "__main__":
    main()
