"""
Text report and matplotlib visualisations for backtest results.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
from .engine import BacktestResult


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def print_report(result: BacktestResult, metrics: Dict[str, Any]) -> None:
    """Print a structured text report to stdout."""
    m = metrics
    days = result.eval_days
    n = len(days)

    sep = "=" * 60
    thin = "-" * 60

    print(sep)
    print("  BACKTEST REPORT")
    print(f"  Eval window : days {days[0]} – {days[-1]}  ({n} days)")
    print(f"  Instruments : {result.nInst}")
    cr, pl_ = result.comm_rates, result.dlr_pos_limits
    print(f"  Commission  : {cr[1]*10000:.1f} bps  (inst 0: {cr[0]*10000:.1f} bps)")
    print(f"  Pos limit   : ${pl_[1]:,.0f} per instrument  (inst 0: ${pl_[0]:,.0f})")
    print(sep)

    print("\n--- COMPETITION SCORE ---")
    print(f"  Score  mu*SR2/(SR2+1)  : {m['score']:>12.4f}  (official 2026 objective)")
    print(f"  Mean daily PL          : {m['mean_pl']:>12.4f}")
    print(f"  StdDev daily PL (pop)  : {m['std_pl']:>12.4f}")
    print(f"  Ann. Sharpe            : {m['ann_sharpe']:>12.4f}")
    print(f"  t-stat (mean PL = 0)   : {m['t_stat']:>12.4f}")
    print(f"  p-value (2-sided)      : {m['p_value']:>12.4f}  (i.i.d. assumption — optimistic)")
    print(f"  PL per $ traded        : {m['pl_per_dollar_traded']:>12.5f}  (eval.py 'ret', not a return)")
    print(f"  Total $ volume         : {m['total_dvolume']:>15,.0f}")
    print(f"  Total commission       : {m['total_commission']:>12.2f}")
    print(f"  Commission drag (bps)  : {m['commission_pct_of_dvolume']*10000:>12.4f}")

    print(f"\n--- DISTRIBUTION & TAIL RISK ---")
    print(f"  Skewness               : {m['skewness']:>12.4f}")
    print(f"  Excess kurtosis        : {m['kurtosis_excess']:>12.4f}  (normal=0, fat tail>0)")
    print(f"  Tail ratio (P95/|P5|)  : {m['tail_ratio']:>12.4f}  (>1 = right-skewed)")
    print(f"  Avg win                : {m['avg_win']:>12.2f}")
    print(f"  Avg loss               : {m['avg_loss']:>12.2f}")
    print(f"  Profit factor          : {m['profit_factor']:>12.4f}  (>1 = profitable)")
    print(f"  Win/loss ratio         : {m['win_loss_ratio']:>12.4f}")
    print(f"  Max consec. losses     : {m['max_consec_loss_days']:>12d} days")
    print(f"  Max consec. wins       : {m['max_consec_win_days']:>12d} days")

    print(f"\n--- DRAWDOWN ANALYSIS ---")
    print(f"  Max drawdown ($)       : {m['max_drawdown']:>12.2f}")
    print(f"  Max DD (% of capital)  : {m['max_drawdown_pct_capital']*100:>12.2f}%  (vs avg gross exposure ${m['capital_base']:,.0f})")
    print(f"  Max PL drawdown ($)    : {m['max_pl_drawdown']:>12.2f}")
    print(f"  Max DD duration        : {m['max_drawdown_duration_days']:>12d} days")
    print(f"  Avg DD duration        : {m['avg_drawdown_duration_days']:>12.1f} days")
    print(f"  % time underwater      : {m['pct_time_underwater']*100:>12.1f}%")
    print(f"  Calmar ratio           : {m['calmar_ratio']:>12.4f}  (ann.PL / max DD)")
    print(f"  Ulcer index            : {m['ulcer_index']:>12.6f}  (RMS % drawdown)")
    print(f"  Pain ratio             : {m['pain_ratio']:>12.4f}")

    print(f"\n--- RISK-ADJUSTED RETURNS ---")
    print(f"  Ann. Sharpe            : {m['ann_sharpe']:>12.4f}")
    print(f"  Sortino ratio          : {m['sortino']:>12.4f}")
    print(f"  Calmar ratio           : {m['calmar_ratio']:>12.4f}")
    print(f"  Omega ratio (MAR=0)    : {m['omega_ratio']:>12.4f}  (>1 = net positive)")
    print(f"  VaR 95%                : {m['var_95']:>12.2f}")
    print(f"  CVaR 95%               : {m['cvar_95']:>12.2f}")
    print(f"  VaR 99%                : {m['var_99']:>12.2f}")
    print(f"  CVaR 99%               : {m['cvar_99']:>12.2f}")
    print(f"  Downside deviation     : {m['downside_dev']:>12.4f}")
    print(f"  Best day               : {m['best_day_pl']:>12.2f}")
    print(f"  Worst day              : {m['worst_day_pl']:>12.2f}")

    print(f"\n--- PL DYNAMICS ---")
    print(f"  Lag-1 autocorrelation  : {m['autocorr_lag1']:>12.4f}  (+= trending, -= mean-rev)")

    print(f"\n--- TRADE DIAGNOSTICS ---")
    print(f"  Hit rate               : {m['hit_rate']*100:>11.1f}%")
    print(f"  Trade frequency        : {m['trade_frequency']*100:>11.1f}% of (day,inst) pairs")
    print(f"  Instruments traded     : {m['n_instruments_traded']:>12d} / {result.nInst}")
    print(f"  Avg holding period     : {m['holding_period_days']:>11.2f} days  (~1/turnover; <1 = intraday churn)")
    print(f"  Avg daily $ volume     : {m['avg_daily_dvolume']:>15,.0f}")
    print(f"  Avg daily turnover     : {m['avg_daily_turnover_ratio']*100:>11.1f}%")
    print(f"  Avg gross exposure     : {m['gross_exposure_avg']:>15,.0f}")
    print(f"  Avg net exposure       : {m['net_exposure_avg']:>15,.0f}")
    print(f"  Clip-binding rate      : {m['clip_binding_pct']*100:>11.1f}% of (day,inst) pairs")
    print(f"  Days with any clipping : {m['n_days_any_clip']:>12d} / {n}")

    ipl = m["inst_pl_total"]
    isharpe = m["inst_sharpe"]
    print(f"\n--- INSTRUMENT ATTRIBUTION ---")
    print(f"  Attribution sum        : ${ipl.sum():>11.2f}  (reconciles to total PL; "
          f"max daily error {m['inst_pl_reconciliation_error']:.2e})")
    print(f"  Top 3 by total PL:")
    for idx in m["top3_instruments"]:
        print(f"    #{idx:2d}  PL=${ipl[idx]:>9.2f}  Sharpe={isharpe[idx]:>7.3f}")
    print(f"  Bottom 3 by total PL:")
    for idx in m["bot3_instruments"]:
        print(f"    #{idx:2d}  PL=${ipl[idx]:>9.2f}  Sharpe={isharpe[idx]:>7.3f}")
    print(f"  Top 3 by Sharpe:")
    for idx in m["top3_sharpe_instruments"]:
        print(f"    #{idx:2d}  PL=${ipl[idx]:>9.2f}  Sharpe={isharpe[idx]:>7.3f}")
    print(f"  Bottom 3 by Sharpe:")
    for idx in m["bot3_sharpe_instruments"]:
        print(f"    #{idx:2d}  PL=${ipl[idx]:>9.2f}  Sharpe={isharpe[idx]:>7.3f}")

    print(sep)


def print_walk_forward_summary(folds: List[Dict[str, Any]]) -> None:
    from .robustness import walk_forward_summary
    s = walk_forward_summary(folds)

    sep = "=" * 60
    print(f"\n{sep}")
    print("  WALK-FORWARD CROSS-VALIDATION RESULTS")
    print(sep)

    # ---- Aggregate summary ----
    scores  = s["scores"]
    sharpes = s["sharpes"]
    means   = np.array([f["mean_pl"]    for f in folds])
    stds    = np.array([f["std_pl"]     for f in folds])
    hits    = np.array([f["hit_rate"]   for f in folds])
    dvols   = np.array([f["total_dvolume"] for f in folds])

    # Pull deeper metrics from stored metrics dicts
    max_dds     = np.array([f["metrics"]["max_drawdown"]          for f in folds])
    sortinos    = np.array([f["metrics"]["sortino"]               for f in folds])
    omegas      = np.array([f["metrics"]["omega_ratio"]           for f in folds])
    pf          = np.array([f["metrics"]["profit_factor"]         for f in folds])
    var95s      = np.array([f["metrics"]["var_95"]                for f in folds])
    cvar95s     = np.array([f["metrics"]["cvar_95"]               for f in folds])
    calmar      = np.array([f["metrics"]["calmar_ratio"]          for f in folds])
    ulcer       = np.array([f["metrics"]["ulcer_index"]           for f in folds])
    autocorrs   = np.array([f["metrics"]["autocorr_lag1"]         for f in folds])
    skews       = np.array([f["metrics"]["skewness"]              for f in folds])
    pct_uw      = np.array([f["metrics"]["pct_time_underwater"]   for f in folds])
    max_streak  = np.array([f["metrics"]["max_consec_loss_days"]  for f in folds])
    turnover    = np.array([f["metrics"]["avg_daily_turnover_ratio"] for f in folds])

    def _row(label, vals, fmt=".4f", pct=False, invert=False):
        """Print mean ± std  [min, max] for a metric array."""
        scale = 100 if pct else 1
        mu  = np.mean(vals) * scale
        std = np.std(vals, ddof=0) * scale
        lo  = np.min(vals) * scale
        hi  = np.max(vals) * scale
        f   = f"{{:>10.{fmt[1:]}}}".format if fmt.startswith(".") else None
        print(f"  {label:<32}  {mu:>10{fmt}}  ±{std:>9{fmt}}  [{lo:>{fmt}}, {hi:>{fmt}}]")

    print("\n  Aggregate across all folds  (mean ± std  [min, max])\n")
    print(f"  {'Metric':<32}  {'Mean':>10}  {'  Std':>10}  {'[Min, Max]':>20}")
    print(f"  {'-'*32}  {'-'*10}  {'-'*10}  {'-'*20}")

    def row(label, vals, fmt=".4f", pct=False):
        scale = 100 if pct else 1
        v = vals * scale
        print(f"  {label:<32}  {np.mean(v):>10{fmt}}  {np.std(v,ddof=0):>10{fmt}}  [{np.min(v):>{fmt}}, {np.max(v):>{fmt}}]")

    row("Score (2026 official)",       scores)
    row("Mean daily PL ($)",           means,   ".4f")
    row("StdDev daily PL ($)",         stds,    ".4f")
    row("Ann. Sharpe",                 sharpes)
    row("Sortino ratio",               sortinos)
    row("Calmar ratio",                calmar)
    row("Omega ratio",                 omegas)
    row("Profit factor",               pf)
    row("Hit rate (%)",                hits,    ".2f", pct=True)
    row("VaR 95% ($)",                 var95s,  ".2f")
    row("CVaR 95% ($)",                cvar95s, ".2f")
    row("Max drawdown ($)",            max_dds, ".2f")
    row("% time underwater",           pct_uw,  ".1f", pct=True)
    row("Ulcer index",                 ulcer,   ".4f")
    row("Skewness",                    skews,   ".4f")
    row("Lag-1 autocorr",             autocorrs,".4f")
    row("Max consec. loss days",       max_streak,".1f")
    row("Avg daily turnover (%)",      turnover, ".1f", pct=True)
    row("Total $ volume",              dvols,   ",.0f")

    # ---- Per-fold detail ----
    print(f"\n\n  Per-fold breakdown\n")
    hdr = (f"  {'Fold':>4}  {'Test window':>14}  {'Score':>8}  {'Sharpe':>7}  "
           f"{'Sortino':>8}  {'Hit%':>6}  {'PF':>6}  {'MaxDD':>9}  "
           f"{'VaR95':>7}  {'%UW':>6}  {'AutoC':>6}  {'MaxLoss':>7}")
    print(hdr)
    print(f"  {'-'*len(hdr.strip())}")
    for i, f in enumerate(folds):
        te = f["test_range"]
        m  = f["metrics"]
        print(
            f"  {f['fold']:>4}  [{te[0]:4d},{te[1]:4d})  "
            f"  {f['score']:>8.3f}"
            f"  {f['ann_sharpe']:>7.3f}"
            f"  {m['sortino']:>8.3f}"
            f"  {m['hit_rate']*100:>6.1f}"
            f"  {m['profit_factor']:>6.3f}"
            f"  {m['max_drawdown']:>9.1f}"
            f"  {m['var_95']:>7.2f}"
            f"  {m['pct_time_underwater']*100:>6.1f}"
            f"  {m['autocorr_lag1']:>6.3f}"
            f"  {m['max_consec_loss_days']:>7d}"
        )

    print(f"\n  Score consistency  : {s['score_positive_frac']*100:.0f}% of folds positive")
    print(f"  Score range        : {s['score_min']:.4f} to {s['score_max']:.4f}")
    score_cv = s['score_std'] / abs(s['score_mean']) if s['score_mean'] != 0 else float('nan')
    print(f"  Score CV (std/|mean|): {score_cv:.3f}  "
          f"(lower = more stable)")
    print(sep)


# ---------------------------------------------------------------------------
# Matplotlib visualisations
# ---------------------------------------------------------------------------

def save_plots(
    result: BacktestResult,
    metrics: Dict[str, Any],
    output_path: str = "backtest_report.pdf",
    wf_folds: Optional[List[Dict[str, Any]]] = None,
    boot_ci: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a multi-page PDF (or PNG bundle) of backtest visualisations.

    Plots generated:
      1.  Equity curve (cumulative portfolio value)
      2.  Daily PL bar chart
      3.  Underwater / drawdown chart
      4.  Daily PL distribution + VaR/CVaR markers
      5.  Rolling 20-day volatility
      6.  Per-instrument PL attribution (horizontal bar)
      7.  Daily turnover ratio
      8.  Rolling 20-day Sharpe ratio
      9.  Per-instrument Sharpe (horizontal bar)
      10. Walk-forward score box plot (if wf_folds provided)
      11. Bootstrap score distribution + CI (if boot_ci provided)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        print("[report] matplotlib not installed — skipping plots.")
        return

    days = result.eval_days
    pl = result.daily_pl
    cum_val = result.cumulative_value
    m = metrics

    # Drawdown series
    running_max = np.maximum.accumulate(cum_val)
    drawdown = cum_val - running_max

    # Colours
    pos_color = "#2ca02c"
    neg_color = "#d62728"
    neutral = "#1f77b4"

    use_pdf = output_path.endswith(".pdf")
    ctx = PdfPages(output_path) if use_pdf else None

    def save_fig(fig: "plt.Figure", name: str) -> None:
        if use_pdf:
            ctx.savefig(fig, bbox_inches="tight")
        else:
            import os
            base = os.path.splitext(output_path)[0]
            fig.savefig(f"{base}_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # --- 1. Equity curve ---
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(days, cum_val, color=neutral, lw=1.5)
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_title("Equity Curve (Cumulative Portfolio Value)")
    ax.set_xlabel("Day"); ax.set_ylabel("Value ($)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "01_equity")

    # --- 2. Daily PL bar chart ---
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = [pos_color if p >= 0 else neg_color for p in pl]
    ax.bar(days, pl, color=colors, width=0.8, alpha=0.85)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Daily PL")
    ax.set_xlabel("Day"); ax.set_ylabel("PL ($)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    save_fig(fig, "02_daily_pl")

    # --- 3. Drawdown / underwater chart ---
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(days, drawdown, 0, color=neg_color, alpha=0.6, label="Drawdown")
    ax.plot(days, drawdown, color=neg_color, lw=0.8)
    ax.set_title("Drawdown ($ from Peak Portfolio Value)")
    ax.set_xlabel("Day"); ax.set_ylabel("Drawdown ($)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "03_drawdown")

    # --- 4. PL distribution ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pl, bins=40, color=neutral, alpha=0.75, edgecolor="white", label="Daily PL")
    ax.axvline(m["var_95"], color="orange", lw=1.5, ls="--", label=f"VaR 95%: {m['var_95']:.1f}")
    ax.axvline(m["cvar_95"], color=neg_color, lw=1.5, ls="--", label=f"CVaR 95%: {m['cvar_95']:.1f}")
    ax.axvline(np.mean(pl), color=pos_color, lw=1.5, ls="-", label=f"Mean: {np.mean(pl):.1f}")
    ax.set_title("Daily PL Distribution")
    ax.set_xlabel("PL ($)"); ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "04_pl_dist")

    # --- 5. Rolling 20-day volatility ---
    roll_vol = m["rolling_vol_20d"]
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(days, roll_vol, color="purple", lw=1.2, label="Rolling 20d vol (PL $)")
    ax.set_title("Rolling 20-Day PL Volatility")
    ax.set_xlabel("Day"); ax.set_ylabel("Std Dev ($)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "05_rolling_vol")

    # --- 6. Per-instrument PL attribution ---
    ipl = m["inst_pl_total"]
    inst_idx = np.argsort(ipl)
    fig, ax = plt.subplots(figsize=(10, 8))
    bar_colors = [pos_color if v >= 0 else neg_color for v in ipl[inst_idx]]
    ax.barh(
        [f"#{i}" for i in inst_idx],
        ipl[inst_idx],
        color=bar_colors,
        alpha=0.85,
        edgecolor="white",
        height=0.7,
    )
    ax.axvline(0, color="black", lw=0.5)
    ax.set_title("Per-Instrument Total PL Attribution")
    ax.set_xlabel("Total PL ($)"); ax.set_ylabel("Instrument")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    save_fig(fig, "06_inst_pnl")

    # --- 7. Daily turnover ratio ---
    turnover = m["turnover_ratio"]
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(days, turnover * 100, color="darkorange", lw=1.0)
    ax.set_title("Daily Turnover Ratio ($ traded / gross exposure)")
    ax.set_xlabel("Day"); ax.set_ylabel("Turnover (%)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "07_turnover")

    # --- 8. Rolling 20-day Sharpe ---
    roll_sharpe = m["rolling_sharpe_20d"]
    fig, ax = plt.subplots(figsize=(12, 3))
    valid = ~np.isnan(roll_sharpe)
    ax.plot(days[valid], roll_sharpe[valid], color="indigo", lw=1.2, label="Rolling 20d Sharpe")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.axhline(1, color=pos_color, lw=0.8, ls=":", label="Sharpe = 1")
    ax.axhline(-1, color=neg_color, lw=0.8, ls=":", label="Sharpe = -1")
    ax.fill_between(days[valid], roll_sharpe[valid], 0,
                    where=roll_sharpe[valid] >= 0, alpha=0.15, color=pos_color)
    ax.fill_between(days[valid], roll_sharpe[valid], 0,
                    where=roll_sharpe[valid] < 0, alpha=0.15, color=neg_color)
    ax.set_title("Rolling 20-Day Annualised Sharpe Ratio")
    ax.set_xlabel("Day"); ax.set_ylabel("Sharpe")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "08_rolling_sharpe")

    # --- 9. Per-instrument Sharpe (horizontal bar) ---
    isharpe = m["inst_sharpe"]
    inst_sidx = np.argsort(isharpe)
    fig, ax = plt.subplots(figsize=(10, 8))
    sharpe_colors = [pos_color if v >= 0 else neg_color for v in isharpe[inst_sidx]]
    ax.barh(
        [f"#{i}" for i in inst_sidx],
        isharpe[inst_sidx],
        color=sharpe_colors,
        alpha=0.85,
        edgecolor="white",
        height=0.7,
    )
    ax.axvline(0, color="black", lw=0.5)
    ax.set_title("Per-Instrument Annualised Sharpe Ratio")
    ax.set_xlabel("Ann. Sharpe"); ax.set_ylabel("Instrument")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    save_fig(fig, "09_inst_sharpe")

    # --- 10. Walk-forward score box plot (optional) ---
    if wf_folds:
        scores = [f["score"] for f in wf_folds]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(scores, vert=True, patch_artist=True,
                   boxprops=dict(facecolor=neutral, alpha=0.6))
        ax.axhline(0, color=neg_color, lw=1, ls="--", label="Score = 0")
        for s in scores:
            ax.scatter([1], [s], color="black", zorder=5, alpha=0.6)
        ax.set_title("Walk-Forward Score Distribution")
        ax.set_ylabel("Score"); ax.set_xticks([])
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        save_fig(fig, "10_wf_scores")

    # --- 11. Bootstrap score CI (optional) ---
    if boot_ci:
        boot_scores = boot_ci["boot_scores"]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(boot_scores, bins=60, color=neutral, alpha=0.75, edgecolor="white")
        ax.axvline(boot_ci["score"], color=pos_color, lw=2, label=f"Point est: {boot_ci['score']:.2f}")
        ax.axvline(boot_ci["ci_lower"], color="orange", lw=1.5, ls="--",
                   label=f"{boot_ci['confidence']*100:.0f}% CI lower: {boot_ci['ci_lower']:.2f}")
        ax.axvline(boot_ci["ci_upper"], color="orange", lw=1.5, ls="--",
                   label=f"{boot_ci['confidence']*100:.0f}% CI upper: {boot_ci['ci_upper']:.2f}")
        ax.set_title("Bootstrap Score Distribution")
        ax.set_xlabel("Score"); ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_fig(fig, "11_bootstrap")

    if use_pdf:
        ctx.close()

    print(f"[report] Plots saved to: {output_path}")
