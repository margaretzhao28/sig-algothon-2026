"""
Parity test: verify that backtest.engine.run_backtest() replicates the
official 2026 eval.py accounting exactly.

Two strategies are checked:
  * flat (zero positions)  — accounting loop sanity: everything must be 0
  * momentum (the starter-code strategy, trades every day) — exercises the
    commission lag, per-instrument rates (inst 0 at 0.2bp), integer clipping
    and the $100k inst-0 limit.

Run with:
    python test_parity.py
"""

import sys
import numpy as np

sys.path.insert(0, ".")
from backtest.data import load_prices, prices_path
from backtest.engine import run_backtest, official_eval_window, NUM_TEST_DAYS
from backtest.metrics import compute_metrics, official_score


def flat_strategy(prices: np.ndarray) -> np.ndarray:
    return np.zeros(prices.shape[0], dtype=int)


def make_momentum_strategy():
    """The algothon26 starter-code strategy (stateful; fresh instance per run)."""
    state = {"pos": None}

    def strategy(prcSoFar: np.ndarray) -> np.ndarray:
        nins, nt = prcSoFar.shape
        if state["pos"] is None:
            state["pos"] = np.zeros(nins)
        if nt < 2:
            return np.zeros(nins)
        lastRet = np.log(prcSoFar[:, -1] / prcSoFar[:, -2])
        lNorm = np.sqrt(lastRet.dot(lastRet))
        lastRet /= lNorm
        rpos = np.array([int(x) for x in 5000 * lastRet / prcSoFar[:, -1]])
        state["pos"] = np.array([int(x) for x in state["pos"] + rpos])
        return state["pos"]

    return strategy


def run_official_loop(prices: np.ndarray, getPosition, numTestDays: int):
    """Replicate the 2026 eval.py calcPL exactly (same ops, same order)."""
    nInst, nt = prices.shape

    commRate = np.full(nInst, 0.0001)
    commRate[0] = 0.00002
    dlrPosLimit = np.full(nInst, 10_000.0)
    dlrPosLimit[0] = 100_000.0

    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    value = 0
    comm = 0

    todayPLL = []
    startDay = nt - numTestDays
    for t in range(startDay, nt + 1):
        prcHistSoFar = prices[:, :t]
        curPrices = prcHistSoFar[:, -1]
        if t < nt:
            newPosOrig = getPosition(prcHistSoFar)
            posLimits = (dlrPosLimit / curPrices).astype(int)
            newPos = np.clip(newPosOrig, -posLimits, posLimits).astype(int)
        else:
            newPos = np.array(curPos)
        deltaPos = newPos - curPos
        cash -= curPrices.dot(deltaPos) + comm
        dvolumes = curPrices * np.abs(deltaPos)
        dvolume = np.sum(dvolumes)
        totDVolume += dvolume
        comm = np.sum(dvolumes * commRate)
        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value
        value = cash + posValue
        if t > startDay:
            todayPLL.append(todayPL)

    pll = np.array(todayPLL)
    plmu, plstd = float(np.mean(pll)), float(np.std(pll))
    ann_sharpe = float(np.sqrt(250) * plmu / plstd) if plstd > 0 else 0.0
    if plmu <= 0 or plstd < 1e-10:
        score = plmu
    else:
        sr = np.sqrt(250) * plmu / plstd
        score = plmu * sr**2 / (sr**2 + 1.0)
    return pll, plmu, plstd, ann_sharpe, score, float(totDVolume)


def check(label: str, prices: np.ndarray, strategy_factory) -> list:
    nt = prices.shape[1]
    num_test = min(NUM_TEST_DAYS, nt - 2)
    eval_start, eval_end = official_eval_window(nt, num_test)

    ref_pll, ref_mu, ref_std, ref_sharpe, ref_score, ref_dvol = run_official_loop(
        prices, strategy_factory(), num_test)

    result = run_backtest(prices, strategy_factory(),
                          eval_start=eval_start, eval_end=eval_end)
    m = compute_metrics(result)

    tol = 1e-9
    failures = []
    print(f"\n[{label}]  window t={eval_start}..{eval_end}  ({num_test} PL days)")
    pll_diff = float(np.max(np.abs(ref_pll - result.daily_pl)))
    status = "PASS" if pll_diff < tol else "FAIL"
    print(f"  {status}  {'daily PL (max abs)':<20} diff={pll_diff:.2e}")
    if status == "FAIL":
        failures.append(f"{label}:daily_pl")

    for name, ref, got in [
        ("mean_pl",       ref_mu,     m["mean_pl"]),
        ("std_pl",        ref_std,    m["std_pl"]),
        ("ann_sharpe",    ref_sharpe, m["ann_sharpe"]),
        ("score",         ref_score,  m["score"]),
        ("total_dvolume", ref_dvol,   m["total_dvolume"]),
    ]:
        diff = abs(ref - got)
        status = "PASS" if diff < tol else "FAIL"
        print(f"  {status}  {name:<20} ref={ref:>14.6f}  got={got:>14.6f}  diff={diff:.2e}")
        if status == "FAIL":
            failures.append(f"{label}:{name}")

    recon = m["inst_pl_reconciliation_error"]
    status = "PASS" if recon < 1e-8 else "FAIL"
    print(f"  {status}  {'inst PL reconciles':<20} max error={recon:.2e}")
    if status == "FAIL":
        failures.append(f"{label}:inst_pl_reconciliation")

    return failures


def test_parity():
    prices = load_prices(prices_path("2026"))

    failures = []
    failures += check("flat", prices, lambda: flat_strategy)
    failures += check("momentum", prices, make_momentum_strategy)

    if failures:
        print(f"\nPARITY TEST FAILED: {failures}")
        sys.exit(1)
    else:
        print("\nAll parity checks passed.")


if __name__ == "__main__":
    test_parity()
