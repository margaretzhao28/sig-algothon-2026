"""
Core simulation engine — byte-for-byte compatible with the 2026 eval.py.

The official 2026 eval.py scores the last `numTestDays = 250` days: with
startDay = nt - numTestDays it iterates t in [startDay, nt], calling
getPosition(prcHist[:, :t]) and trading at column t-1's prices on every t < nt.
The final iteration t = nt is mark-only (no trading).  Two accounting quirks
faithfully replicated here:

  * Commission lag — fees for day t's trades are deducted from cash at
    iteration t+1 (`cash -= curPrices.dot(deltaPos) + comm` uses the previous
    iteration's comm).  Consequently the first trading day's PL is recorded on
    the following iteration, and the mark-only day still pays the final
    trading day's fees.
  * Per-instrument rates — commission is 1bp of gross dollar volume except
    instrument 0 at 0.2bp; position limits are $10k except instrument 0 at
    $100k.  Positions are clipped to int($limit / price) and truncated to
    integer shares.

PL is recorded for t in (eval_start, eval_end], i.e. n = eval_end - eval_start
days.  Row i of every per-day array corresponds to: the trade executed at
iteration eval_start + i (at prices_prev[i], column eval_start+i-1) and the PL
that trade's position earned when marked at prices[i] (column eval_start+i),
minus that trade's commission.  This alignment makes per-instrument
attribution exact: pl[i] == sum_j pos[i,j]*(prices[i,j]-prices_prev[i,j]) -
commission_inst[i,j].

For everyday use, run_backtest() takes a RESEARCH_FRAC: the fraction of days
the strategy was designed/tuned on. There is no separate "holdout" concept —
since getMyPosition always sees full price history (the framework can't sandbox
a training step), the only thing that makes a day out-of-sample is that you
didn't look at it while building the strategy. So:
  - the strategy is "researched" on [0, research_frac * nt)
  - by default, it's *evaluated* on the rest: [research_frac * nt, nt)
  - explicit eval_start/eval_end override this entirely (e.g. to test on a
    sub-window and deliberately leave the remainder untouched — that
    discipline is the user's responsibility, the framework doesn't enforce it)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np

NUM_TEST_DAYS: int = 250          # official 2026 eval window: the last 250 days
COMM_RATE: float = 0.0001         # 1bp default commission
INST0_COMM_RATE: float = 0.00002  # 0.2bp for instrument 0
DLR_POS_LIMIT: float = 10_000.0   # default per-instrument dollar limit
INST0_DLR_POS_LIMIT: float = 100_000.0  # $100k for instrument 0
RESEARCH_FRAC: float = 0.8   # fraction of days available for research/design by default


def official_eval_window(nt: int, num_test_days: int = NUM_TEST_DAYS) -> tuple[int, int]:
    """The official 2026 window: score the last `num_test_days` days."""
    return nt - num_test_days, nt


@dataclass
class BacktestResult:
    """All per-day simulation output needed for metrics and reports."""

    # 1-D arrays of length n_eval_days
    daily_pl: np.ndarray           # PnL recorded on each evaluated day
    cumulative_value: np.ndarray   # running portfolio value (cash + positions)
    daily_dvolume: np.ndarray      # gross dollar volume of row i's trade
    daily_commission: np.ndarray   # commission on row i's trade (hits row i's PL)

    # 2-D arrays of shape (n_eval_days, nInst)
    positions_requested: np.ndarray  # raw output of getMyPosition each day
    positions_held: np.ndarray       # after clipping to dollar limits
    prices: np.ndarray               # mark prices for row i's PL
    prices_prev: np.ndarray          # trade prices for row i's position
    delta_positions: np.ndarray      # position changes (shares bought/sold)
    daily_commission_inst: np.ndarray  # per-instrument commission per row

    # Scalar summary (mirrors eval.py output)
    total_dvolume: float
    final_return: float              # value / total_dvolume

    # Metadata
    eval_days: np.ndarray            # trading-day indices (the t values traded on)
    nInst: int
    comm_rates: np.ndarray           # (nInst,) commission rate per instrument
    dlr_pos_limits: np.ndarray       # (nInst,) dollar position limit per instrument

    # Optional strategy-reported "fair price" series — purely a research/
    # visualization side-channel (see fair_price_fn below). Dict of
    # series name -> (n_eval_days, nInst) array, NaN where the strategy
    # didn't report a value for that instrument/day.
    fair_price: Optional[dict[str, np.ndarray]] = None


def default_eval_window(nt: int, research_frac: float = RESEARCH_FRAC) -> tuple[int, int]:
    """
    Derive the default (eval_start, eval_end) as "everything after research".

    research_frac * nt days are reserved for research/design; the remainder
    [research_end, nt) is the default evaluation window.

    For 500 days at research_frac=0.5:  eval_start = 250,  eval_end = 500 —
    which coincides with the official numTestDays=250 window on the 2026 data.
    """
    research_end = max(1, min(int(round(nt * research_frac)), nt - 1))
    return research_end, nt


def _per_instrument(value, inst0_value, nInst: int) -> np.ndarray:
    """Broadcast a scalar (with an instrument-0 override) or pass an array through."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.full(nInst, float(value))
        if inst0_value is not None:
            arr[0] = float(inst0_value)
    return arr


def run_backtest(
    prices: np.ndarray,
    strategy: Callable[[np.ndarray], np.ndarray],
    *,
    eval_start: Optional[int] = None,
    eval_end: Optional[int] = None,
    research_frac: float = RESEARCH_FRAC,
    comm_rate: float | np.ndarray = COMM_RATE,
    inst0_comm_rate: Optional[float] = INST0_COMM_RATE,
    dlr_pos_limit: float | np.ndarray = DLR_POS_LIMIT,
    inst0_dlr_pos_limit: Optional[float] = INST0_DLR_POS_LIMIT,
    verbose: bool = False,
    fair_price_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> BacktestResult:
    """
    Run one backtest pass over the specified day range (2026 accounting).

    Parameters
    ----------
    prices       : (nInst, nt) price array — the full history available
    strategy     : callable matching getMyPosition(prcHist: ndarray) -> ndarray
    eval_start   : first trading day t (strategy sees prices[:, :t]); its PL is
                   recorded on the next iteration.  If None, derived from
                   research_frac.
    eval_end     : last trading day is eval_end - 1; iteration t = eval_end is
                   mark-only.  Must satisfy eval_end <= nt.  If None, nt.
    research_frac: fraction of days reserved for research/design, used to derive
                   the default eval window [research_frac * nt, nt).
    comm_rate    : commission rate on gross dollar volume (scalar or (nInst,)
                   array; default 1bp)
    inst0_comm_rate : instrument-0 override when comm_rate is scalar
                   (default 0.2bp; pass None to disable)
    dlr_pos_limit: max dollar exposure per instrument (scalar or array; $10k)
    inst0_dlr_pos_limit : instrument-0 override when dlr_pos_limit is scalar
                   (default $100k; pass None to disable)
    verbose      : print per-day output matching eval.py format
    fair_price_fn: optional callable(prcHist) -> (nInst,) array OR a dict of
                   {series_name: (nInst,) array} for multiple lines (e.g.
                   {"fast MA": ..., "slow MA": ...}), called alongside `strategy`
                   on every trading day purely for visualization/research. Never
                   touches positions/PL/accounting, so it cannot affect the score —
                   it's a read-only side channel. Use NaN for instruments with no
                   fair-price opinion.

    Returns
    -------
    BacktestResult with per-day arrays and summary scalars.
    """
    nInst, nt = prices.shape

    if eval_start is None or eval_end is None:
        _start, _end = default_eval_window(nt, research_frac)
        if eval_start is None:
            eval_start = _start
        if eval_end is None:
            eval_end = _end

    eval_start = int(eval_start)
    eval_end = int(min(eval_end, nt))
    if eval_start < 1:
        raise ValueError("eval_start must be >= 1 (the strategy needs at least one price)")
    if eval_end <= eval_start:
        raise ValueError("eval_end must be > eval_start")

    comm_rates = _per_instrument(comm_rate, inst0_comm_rate, nInst)
    dlr_pos_limits = _per_instrument(dlr_pos_limit, inst0_dlr_pos_limit, nInst)

    n = eval_end - eval_start
    eval_days = np.arange(eval_start, eval_end, dtype=int)

    daily_pl = np.zeros(n)
    cumulative_value = np.zeros(n)
    daily_dvolume = np.zeros(n)
    daily_commission = np.zeros(n)
    positions_requested = np.zeros((n, nInst))
    positions_held = np.zeros((n, nInst))
    prices_out = np.zeros((n, nInst))
    prices_prev = np.zeros((n, nInst))
    delta_positions = np.zeros((n, nInst))
    daily_commission_inst = np.zeros((n, nInst))
    fair_price: Optional[dict] = {} if fair_price_fn is not None else None

    # State — mirrors eval.py calcPL exactly, including the commission lag:
    # `comm` computed at iteration t is deducted from cash at iteration t+1.
    cash = 0.0
    cur_pos = np.zeros(nInst)
    tot_dvolume = 0.0
    value = 0.0
    comm = 0.0

    for t in range(eval_start, eval_end + 1):
        prc_hist_so_far = prices[:, :t]
        cur_prices = prc_hist_so_far[:, -1]

        if t < eval_end:
            new_pos_orig = strategy(prc_hist_so_far)
            pos_limits = (dlr_pos_limits / cur_prices).astype(int)
            new_pos = np.clip(new_pos_orig, -pos_limits, pos_limits).astype(int)
        else:
            new_pos = np.array(cur_pos)

        delta = new_pos - cur_pos
        cash -= cur_prices.dot(delta) + comm

        dvolumes = cur_prices * np.abs(delta)
        dvolume = np.sum(dvolumes)
        tot_dvolume += dvolume
        comm_inst = dvolumes * comm_rates
        comm = np.sum(comm_inst)

        cur_pos = np.array(new_pos)
        pos_value = cur_pos.dot(cur_prices)
        today_pl = cash + pos_value - value
        value = cash + pos_value
        ret = value / tot_dvolume if tot_dvolume > 0 else 0.0

        if t < eval_end:
            # trading-day row: the position established here earns row i's PL
            i = t - eval_start
            positions_requested[i] = new_pos_orig
            positions_held[i] = new_pos
            prices_prev[i] = cur_prices
            delta_positions[i] = delta
            daily_dvolume[i] = dvolume
            daily_commission[i] = comm
            daily_commission_inst[i] = comm_inst
            if fair_price_fn is not None:
                fp = fair_price_fn(prc_hist_so_far)
                if not isinstance(fp, dict):
                    fp = {"fair": fp}
                for key, vals in fp.items():
                    if key not in fair_price:
                        fair_price[key] = np.full((n, nInst), np.nan)
                    fair_price[key][i] = vals

        if t > eval_start:
            # PL-recording row: marks row i's position at today's prices
            i = t - eval_start - 1
            daily_pl[i] = today_pl
            cumulative_value[i] = value
            prices_out[i] = cur_prices

        if verbose:
            print(
                f"Day {t:4d}  value: {value:12.2f}  todayPL: ${today_pl:10.2f}"
                f"  $-traded: {tot_dvolume:12.0f}  return: {ret:.5f}"
            )

    final_ret = value / tot_dvolume if tot_dvolume > 0 else 0.0

    return BacktestResult(
        daily_pl=daily_pl,
        cumulative_value=cumulative_value,
        daily_dvolume=daily_dvolume,
        daily_commission=daily_commission,
        positions_requested=positions_requested,
        positions_held=positions_held,
        prices=prices_out,
        prices_prev=prices_prev,
        delta_positions=delta_positions,
        daily_commission_inst=daily_commission_inst,
        total_dvolume=float(tot_dvolume),
        final_return=float(final_ret),
        eval_days=eval_days,
        nInst=nInst,
        comm_rates=comm_rates,
        dlr_pos_limits=dlr_pos_limits,
        fair_price=fair_price,
    )
