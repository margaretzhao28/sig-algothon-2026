"""
All evaluation metrics — competition core, risk/drawdown, and trade diagnostics.

All functions accept a BacktestResult and return plain Python dicts so they
can be logged, serialised, or aggregated without coupling to any display layer.
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np
from scipy import stats
from .engine import BacktestResult

# Days per trading year — used for annualisation (matches eval.py's sqrt(250)).
ANN_FACTOR = 250


# ---------------------------------------------------------------------------
# Competition score (matches the 2026 eval.py exactly — ddof=0, param=1)
# ---------------------------------------------------------------------------

def official_score(pl: np.ndarray) -> float:
    """2026 competition objective: mu * SR^2/(SR^2 + 1) for mu > 0, else mu."""
    mu = float(np.mean(pl))
    sigma = float(np.std(pl, ddof=0))
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = np.sqrt(ANN_FACTOR) * mu / sigma
    return mu * sr**2 / (sr**2 + 1.0)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def _core_metrics(r: BacktestResult) -> Dict[str, Any]:
    pl = r.daily_pl
    n = len(pl)
    plmu = float(np.mean(pl))
    plstd = float(np.std(pl, ddof=0))             # population std — matches eval.py score
    plstd_sample = float(np.std(pl, ddof=1)) if n > 1 else 0.0  # sample std for inference
    ann_sharpe = float(np.sqrt(ANN_FACTOR) * plmu / plstd) if plstd > 0 else 0.0
    score = official_score(pl)                    # 2026 formula

    # Inferential statistics on mean daily PL (is the edge distinguishable from zero?).
    # Uses the sample standard error; df = n-1.  NOTE: assumes daily PL is approximately
    # i.i.d. — serial correlation (see autocorr_lag1) inflates significance, so treat the
    # p-value as optimistic.  The block bootstrap CI is the more robust check.
    if n > 1 and plstd_sample > 0:
        se = plstd_sample / np.sqrt(n)
        t_stat = plmu / se
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df=n - 1))
    else:
        t_stat = 0.0
        p_value = 1.0

    return {
        "mean_pl": plmu,
        "std_pl": plstd,
        "std_pl_sample": plstd_sample,
        "score": score,
        "ann_sharpe": ann_sharpe,
        "t_stat": float(t_stat),
        "p_value": p_value,
        "total_dvolume": r.total_dvolume,
        "total_commission": float(np.sum(r.daily_commission)),
        "commission_pct_of_dvolume": float(np.sum(r.daily_commission) / r.total_dvolume)
        if r.total_dvolume > 0
        else 0.0,
        # eval.py's "ret": cumulative PL per dollar traded — NOT a return on capital.
        "pl_per_dollar_traded": r.final_return,
        "n_eval_days": n,
    }


# ---------------------------------------------------------------------------
# Risk and drawdown
# ---------------------------------------------------------------------------

def _risk_metrics(r: BacktestResult) -> Dict[str, Any]:
    pl = r.daily_pl
    cum = r.cumulative_value

    # Drawdown on cumulative PL curve.  The equity path starts at 0 (no seed capital),
    # so a percentage drawdown relative to the running peak is undefined/meaningless
    # (the peak is ~0 early on).  We therefore report drawdown in DOLLARS, and express
    # the percentage relative to the average capital actually deployed (mean gross
    # exposure) — an interpretable, stable denominator for a dollar-PL strategy.
    running_max = np.maximum.accumulate(cum)
    drawdown = cum - running_max          # negative or zero
    max_drawdown = float(np.min(drawdown))

    prices = r.prices
    pos = r.positions_held
    gross_exposure_series = (np.abs(pos) * prices).sum(axis=1)
    capital_base = float(np.mean(gross_exposure_series)) if len(gross_exposure_series) else 0.0
    max_drawdown_pct_capital = (
        float(max_drawdown / capital_base) if capital_base > 0 else 0.0
    )

    # Drawdown on PL series (identical to value path here; kept for clarity)
    cum_pl = np.cumsum(pl)
    pl_running_max = np.maximum.accumulate(cum_pl)
    pl_drawdown = cum_pl - pl_running_max
    max_pl_drawdown = float(np.min(pl_drawdown))

    # VaR and CVaR at 95% and 99% (historical, on daily PL)
    var_95 = float(np.percentile(pl, 5))
    cvar_95 = float(np.mean(pl[pl <= var_95]))
    var_99 = float(np.percentile(pl, 1))
    cvar_99 = float(np.mean(pl[pl <= var_99])) if np.any(pl <= var_99) else var_99

    # Sortino — target downside deviation uses the FULL sample size N in the denominator
    # (standard Sortino & Price definition), not just the count of down days.  Dividing
    # by the number of negative days (a common error) overstates downside deviation.
    downside_sq = np.minimum(pl, 0.0) ** 2
    downside_dev = float(np.sqrt(np.mean(downside_sq)))
    sortino = float(np.mean(pl) / downside_dev * np.sqrt(ANN_FACTOR)) if downside_dev > 0 else 0.0

    # Rolling 20-day volatility
    rolling_vol = np.array([
        np.std(pl[max(0, i - 19):i + 1], ddof=0)
        for i in range(len(pl))
    ])

    # Per-instrument PL attribution (exact — reconciles to total daily PL).
    inst_pl = _instrument_pl(r)
    inst_pl_total = inst_pl.sum(axis=0)           # (nInst,) total PL per instrument
    top3_inst = np.argsort(inst_pl_total)[::-1][:3].tolist()
    bot3_inst = np.argsort(inst_pl_total)[:3].tolist()
    # Reconciliation: sum over instruments of per-day PL must equal total daily PL.
    recon_error = float(np.max(np.abs(inst_pl.sum(axis=1) - pl)))

    return {
        "max_drawdown": max_drawdown,
        "max_drawdown_pct_capital": max_drawdown_pct_capital,
        "capital_base": capital_base,
        "max_pl_drawdown": max_pl_drawdown,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "var_99": var_99,
        "cvar_99": cvar_99,
        "sortino": sortino,
        "downside_dev": downside_dev,
        "best_day_pl": float(np.max(pl)),
        "worst_day_pl": float(np.min(pl)),
        "rolling_vol_20d": rolling_vol,          # array — used by report/plots
        "inst_pl_total": inst_pl_total,          # array — (nInst,)
        "inst_pl_reconciliation_error": recon_error,
        "top3_instruments": top3_inst,
        "bot3_instruments": bot3_inst,
    }


def _instrument_pl(r: BacktestResult) -> np.ndarray:
    """
    Exact per-instrument, per-day PL.  Shape (n_days, nInst).

    The engine aligns each row i so that positions_held[i] was traded at
    prices_prev[i] and marked at prices[i], with that trade's commission in
    daily_commission_inst[i] (the 2026 eval charges day t's fees at t+1, which
    lands in the same recorded-PL row).  So the decomposition is uniform:

        ipl[i, j] = pos_held[i, j] * (price[i, j] - price_prev[i, j])
                    - commission_inst[i, j]

    By construction the row sums equal r.daily_pl exactly (verified by
    inst_pl_reconciliation_error ~ 0).
    """
    ipl = r.positions_held * (r.prices - r.prices_prev)   # mark-to-market
    ipl -= r.daily_commission_inst                        # per-instrument commission
    return ipl


# ---------------------------------------------------------------------------
# Trade / execution diagnostics
# ---------------------------------------------------------------------------

def _trade_metrics(r: BacktestResult) -> Dict[str, Any]:
    pos = r.positions_held        # (n_days, nInst)
    delta = r.delta_positions     # (n_days, nInst)
    prices = r.prices             # (n_days, nInst)
    pl = r.daily_pl
    n = len(pl)

    # Turnover: gross dollar traded per day / portfolio value
    portfolio_value = np.abs(pos * prices).sum(axis=1)  # gross exposure each day
    with np.errstate(invalid="ignore", divide="ignore"):
        turnover_ratio = np.where(
            portfolio_value > 0,
            r.daily_dvolume / portfolio_value,
            0.0,
        )

    # Trade frequency: fraction of (day, inst) pairs with non-zero delta
    total_slots = n * r.nInst
    trades_mask = delta != 0
    trade_frequency = float(trades_mask.sum() / total_slots) if total_slots > 0 else 0.0
    n_instruments_traded = int(np.any(delta != 0, axis=0).sum())

    # Hit rate: fraction of days with positive PL
    hit_rate = float(np.mean(pl > 0))

    # Holding period proxy: avg abs position / avg daily shares traded
    avg_abs_pos = float(np.mean(np.abs(pos)))
    avg_daily_shares = float(np.mean(np.abs(delta).sum(axis=1)))
    holding_period = avg_abs_pos / avg_daily_shares if avg_daily_shares > 0 else np.nan

    # Position-limit binding: how often clip changed the requested position
    clipped_mask = r.positions_requested != r.positions_held
    clip_binding_pct = float(clipped_mask.sum() / total_slots) if total_slots > 0 else 0.0
    n_days_clipped = int(np.any(clipped_mask, axis=1).sum())

    # Gross and net exposure (dollar)
    gross_exposure = float(np.mean((np.abs(pos) * prices).sum(axis=1)))
    net_exposure = float(np.mean((pos * prices).sum(axis=1)))

    # Avg daily dvolume
    avg_daily_dvolume = float(np.mean(r.daily_dvolume))

    return {
        "trade_frequency": trade_frequency,
        "n_instruments_traded": n_instruments_traded,
        "hit_rate": hit_rate,
        "holding_period_days": holding_period,
        "avg_daily_dvolume": avg_daily_dvolume,
        "avg_daily_turnover_ratio": float(np.mean(turnover_ratio)),
        "gross_exposure_avg": gross_exposure,
        "net_exposure_avg": net_exposure,
        "clip_binding_pct": clip_binding_pct,
        "n_days_any_clip": n_days_clipped,
        "turnover_ratio": turnover_ratio,   # array — used by plots
    }


# ---------------------------------------------------------------------------
# Advanced / distribution metrics
# ---------------------------------------------------------------------------

def _advanced_metrics(r: BacktestResult) -> Dict[str, Any]:
    pl = r.daily_pl
    cum = r.cumulative_value
    n = len(pl)

    # --- Distribution shape ---
    plmu = float(np.mean(pl))
    plstd = float(np.std(pl, ddof=0))
    skewness = float(_skew(pl))
    kurtosis = float(_kurt(pl))          # excess kurtosis (normal = 0)
    p95 = float(np.percentile(pl, 95))
    p5  = float(np.percentile(pl, 5))
    tail_ratio = float(p95 / abs(p5)) if abs(p5) > 1e-10 else np.nan

    # --- Win / loss analysis ---
    wins  = pl[pl > 0]
    losses = pl[pl < 0]
    avg_win  = float(np.mean(wins))  if len(wins)  > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    profit_factor = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else np.nan
    win_loss_ratio = float(avg_win / abs(avg_loss)) if avg_loss != 0 else np.nan

    # --- Streak analysis ---
    max_consec_loss, max_consec_win = _streak_counts(pl)

    # --- Drawdown duration ---
    running_max = np.maximum.accumulate(cum)
    in_dd = (cum < running_max)                     # True on days that are underwater
    pct_time_underwater = float(np.mean(in_dd))
    max_dd_duration, avg_dd_duration = _drawdown_durations(in_dd)

    # --- Calmar ratio ---
    max_dd_abs = float(np.min(cum - running_max))
    total_pl = float(cum[-1] - (cum[0] - pl[0]))    # total PL over eval window
    ann_pl = total_pl * 250 / n
    calmar = float(ann_pl / abs(max_dd_abs)) if max_dd_abs < 0 else np.nan

    # --- Ulcer index and Pain ratio ---
    with np.errstate(invalid="ignore", divide="ignore"):
        dd_pct = np.where(running_max != 0, (cum - running_max) / running_max, 0.0)
    ulcer_index = float(np.sqrt(np.mean(dd_pct ** 2)))
    pain_ratio = float((plmu / plstd) / ulcer_index) if ulcer_index > 0 else np.nan

    # --- Omega ratio (MAR = 0) ---
    gains  = np.sum(np.maximum(pl, 0))
    losssum = np.sum(np.maximum(-pl, 0))
    omega = float(gains / losssum) if losssum > 0 else np.nan

    # --- Lag-1 PL autocorrelation ---
    if n > 2 and np.std(pl[:-1]) > 0 and np.std(pl[1:]) > 0:
        autocorr_lag1 = float(np.corrcoef(pl[:-1], pl[1:])[0, 1])
    else:
        autocorr_lag1 = np.nan

    # --- Rolling 20-day Sharpe ---
    rolling_sharpe = np.full(n, np.nan)
    for i in range(19, n):
        w = pl[i - 19:i + 1]
        std_w = np.std(w, ddof=0)
        rolling_sharpe[i] = float(np.sqrt(ANN_FACTOR) * np.mean(w) / std_w) if std_w > 0 else 0.0

    # --- Per-instrument Sharpe ---
    inst_pl_arr = _instrument_pl(r)                 # (n_days, nInst)
    inst_mu  = inst_pl_arr.mean(axis=0)
    inst_std = inst_pl_arr.std(axis=0, ddof=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        inst_sharpe = np.where(inst_std > 0, np.sqrt(ANN_FACTOR) * inst_mu / inst_std, 0.0)
    top3_sharpe = np.argsort(inst_sharpe)[::-1][:3].tolist()
    bot3_sharpe = np.argsort(inst_sharpe)[:3].tolist()

    return {
        # Distribution
        "skewness": skewness,
        "kurtosis_excess": kurtosis,
        "tail_ratio": tail_ratio,
        # Win/loss
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "win_loss_ratio": win_loss_ratio,
        # Streaks
        "max_consec_loss_days": max_consec_loss,
        "max_consec_win_days": max_consec_win,
        # Drawdown duration
        "pct_time_underwater": pct_time_underwater,
        "max_drawdown_duration_days": max_dd_duration,
        "avg_drawdown_duration_days": avg_dd_duration,
        # Risk-adjusted
        "calmar_ratio": calmar,
        "ulcer_index": ulcer_index,
        "pain_ratio": pain_ratio,
        "omega_ratio": omega,
        # PL dynamics
        "autocorr_lag1": autocorr_lag1,
        # Arrays for plots
        "rolling_sharpe_20d": rolling_sharpe,     # (n_days,) — NaN for first 19 days
        "inst_sharpe": inst_sharpe,               # (nInst,)
        "top3_sharpe_instruments": top3_sharpe,
        "bot3_sharpe_instruments": bot3_sharpe,
    }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _skew(x: np.ndarray) -> float:
    mu = np.mean(x)
    std = np.std(x, ddof=0)
    if std < 1e-10:
        return 0.0
    return float(np.mean(((x - mu) / std) ** 3))


def _kurt(x: np.ndarray) -> float:
    """Excess kurtosis (Fisher definition, normal = 0)."""
    mu = np.mean(x)
    std = np.std(x, ddof=0)
    if std < 1e-10:
        return 0.0
    return float(np.mean(((x - mu) / std) ** 4) - 3.0)


def _streak_counts(pl: np.ndarray) -> tuple[int, int]:
    """Return (max_consec_losses, max_consec_wins)."""
    max_loss = max_win = cur_loss = cur_win = 0
    for p in pl:
        if p < 0:
            cur_loss += 1
            cur_win = 0
        elif p > 0:
            cur_win += 1
            cur_loss = 0
        else:
            cur_loss = cur_win = 0
        max_loss = max(max_loss, cur_loss)
        max_win  = max(max_win,  cur_win)
    return max_loss, max_win


def _drawdown_durations(in_dd: np.ndarray) -> tuple[int, float]:
    """
    Given a boolean array (True = underwater), return
    (max_drawdown_duration_days, avg_drawdown_duration_days).
    """
    durations = []
    length = 0
    for flag in in_dd:
        if flag:
            length += 1
        else:
            if length > 0:
                durations.append(length)
            length = 0
    if length > 0:
        durations.append(length)
    if not durations:
        return 0, 0.0
    return int(max(durations)), float(np.mean(durations))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_metrics(r: BacktestResult) -> Dict[str, Any]:
    """Compute all metric groups and return a flat dict."""
    m: Dict[str, Any] = {}
    m.update(_core_metrics(r))
    m.update(_risk_metrics(r))
    m.update(_trade_metrics(r))
    m.update(_advanced_metrics(r))
    return m
