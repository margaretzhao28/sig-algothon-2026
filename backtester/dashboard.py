"""
Interactive backtest dashboard.

    .venv/bin/streamlit run dashboard.py

Two ways to get a result on screen:
  * "Last backtest run" — shows whatever run_backtest.py last produced
    (it saves results/last_run.pkl on every run and auto-launches this app).
    Hit R (rerun) in the browser after a new backtest to refresh.
  * "Run a strategy" — pick any strategy module in this directory and run it
    live from the sidebar.

Panels share one time axis: prices (+ buy/sell markers), PnL, positions.
"""

from __future__ import annotations
import glob
import importlib
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

BACKTESTER_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BACKTESTER_DIR)
SRC_STRATEGIES_DIR = os.path.join(REPO_ROOT, "src", "strategies")

sys.path.insert(0, BACKTESTER_DIR)
if SRC_STRATEGIES_DIR not in sys.path:
    sys.path.insert(0, SRC_STRATEGIES_DIR)

from backtest.data import load_prices, prices_path, ACTIVE_DATASET
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics
from backtest.visualize import build_figure, load_instrument_names

LAST_RUN_PKL = os.path.join("results", "last_run.pkl")


# ---------------------------------------------------------------------------
# Data access (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def _load_prices(dataset: str) -> np.ndarray:
    return load_prices(prices_path(dataset))


@st.cache_data
def _load_last_run(mtime: float) -> dict:
    """mtime busts the cache whenever run_backtest.py writes a new pickle."""
    with open(LAST_RUN_PKL, "rb") as f:
        return pickle.load(f)


def _strategy_modules() -> list[str]:
    """Importable strategies from backtester/ and src/strategies/."""
    mods = set()
    for folder in (BACKTESTER_DIR, SRC_STRATEGIES_DIR):
        for path in glob.glob(os.path.join(folder, "*.py")):
            name = os.path.basename(path)[:-3]
            if name in ("dashboard", "eval", "run_backtest", "test_parity"):
                continue
            try:
                with open(path) as f:
                    if "def getMyPosition" in f.read():
                        mods.add(name)
            except OSError:
                continue
    return sorted(mods)


@st.cache_data
def _run_strategy(strategy_module: str, dataset: str, research_frac: float,
                  _cache_salt: float) -> dict:
    prices = _load_prices(dataset)
    mod = importlib.import_module(strategy_module)
    importlib.reload(mod)  # reset any global state (e.g. currentPos)
    result = run_backtest(prices, mod.getMyPosition, research_frac=research_frac)
    metrics = compute_metrics(result)
    try:
        from backtest.visualize import ridge_rank_annotations
        pos_annot = ridge_rank_annotations(prices, result.eval_days, mod)
    except Exception:
        pos_annot = None
    return {
        "result": result,
        "metrics": metrics,
        "strategy": strategy_module,
        "dataset": dataset,
        "prices_file": prices_path(dataset),
        "pos_annot": pos_annot,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Backtest Dashboard", layout="wide", page_icon="📈")

    with st.sidebar:
        st.header("⚙️ Source")
        has_last = os.path.exists(LAST_RUN_PKL)
        mode = st.radio(
            "Result to display",
            ["Last backtest run", "Run a strategy"],
            index=0 if has_last else 1,
        )

        run = None
        if mode == "Last backtest run":
            if not has_last:
                st.warning("No saved run yet — do `python run_backtest.py ...` first, "
                           "or switch to 'Run a strategy'.")
                return
            run = _load_last_run(os.path.getmtime(LAST_RUN_PKL))
            st.caption(f"Loaded {LAST_RUN_PKL}\n\n"
                       f"**{run['strategy']}** on **{run['dataset']}** at {run['timestamp']}")
            if st.button("🔄 Reload latest"):
                st.cache_data.clear()
                st.rerun()
        else:
            mods = _strategy_modules()
            if not mods:
                st.error("No strategy modules found (need a *.py with getMyPosition).")
                return
            strategy_module = st.selectbox("Strategy module", mods,
                                           index=mods.index("anso_mm") if "anso_mm" in mods else 0)
            datasets = sorted(os.path.basename(d) for d in glob.glob("data/*") if os.path.isdir(d))
            dataset = st.selectbox("Dataset", datasets,
                                   index=datasets.index(ACTIVE_DATASET) if ACTIVE_DATASET in datasets else 0)
            research_frac = st.slider("Research fraction (eval = the rest)", 0.1, 0.9, 0.7, 0.05)
            if st.button("▶ Run backtest", type="primary"):
                st.session_state["live_salt"] = datetime.now().timestamp()
            if "live_salt" in st.session_state:
                run = _run_strategy(strategy_module, dataset, research_frac,
                                    st.session_state["live_salt"])
            else:
                st.info("Press ▶ Run backtest")
                return

    result = run["result"]
    m = run["metrics"]
    names = load_instrument_names(run.get("prices_file", prices_path(run["dataset"]))) \
        or [f"Inst_{j:02d}" for j in range(result.nInst)]

    # ---- Header metrics ----------------------------------------------------
    st.title(f"📈 {run['strategy']} — {run['dataset']}")
    days = result.eval_days
    st.caption(f"Eval days {days[0]}–{days[-1]}  ({len(days)} days)  ·  "
               f"commission 1bp (inst 0: 0.2bp)  ·  run at {run['timestamp']}")

    c = st.columns(6)
    c[0].metric("Score (2026)", f"{m['score']:.3f}")
    c[1].metric("Mean daily PL", f"${m['mean_pl']:.2f}")
    c[2].metric("Ann. Sharpe", f"{m['ann_sharpe']:.2f}")
    c[3].metric("Total PL", f"${result.cumulative_value[-1]:,.0f}")
    c[4].metric("Max drawdown", f"${m['max_drawdown']:,.0f}")
    c[5].metric("Hit rate", f"{m['hit_rate']*100:.0f}%")

    # ---- Plot controls -----------------------------------------------------
    traded = np.where(np.abs(result.delta_positions).sum(axis=0) > 0)[0]
    traded_names = [names[j] for j in traded]

    ctl = st.columns([3, 1, 1])
    with ctl[0]:
        chosen = st.multiselect(
            "Instruments on the chart",
            options=list(names),
            default=traded_names if traded_names else list(names)[:5],
        )
    with ctl[1]:
        show_markers = st.checkbox("Buy/sell markers", value=True)
    with ctl[2]:
        normalize = st.checkbox("Normalize prices", value=True)

    idx = [names.index(nm) for nm in chosen]
    if not idx:
        st.warning("Pick at least one instrument.")
        return

    fig = build_figure(
        result, instrument_names=names, title="",
        instruments=idx, show_markers=show_markers, normalize=normalize,
        pos_annot=run.get("pos_annot"),
    )
    fig.update_layout(height=850, margin=dict(t=40))
    st.plotly_chart(fig, width="stretch")

    # ---- Per-instrument table ----------------------------------------------
    with st.expander("📊 Per-instrument breakdown"):
        ipl = m["inst_pl_total"]
        isharpe = m["inst_sharpe"]
        n_trades = (result.delta_positions != 0).sum(axis=0)
        dvol = (np.abs(result.delta_positions) * result.prices_prev).sum(axis=0)
        df = pd.DataFrame({
            "Instrument": names,
            "Total PL ($)": np.round(ipl, 2),
            "Sharpe": np.round(isharpe, 3),
            "Trades": n_trades,
            "$ Volume": np.round(dvol, 0),
        })
        df = df[df["Trades"] > 0].sort_values("Total PL ($)", ascending=False)
        if df.empty:
            st.info("No trades in this run.")
        else:
            st.dataframe(df, width="stretch", hide_index=True)

    # ---- Daily PL distribution ---------------------------------------------
    with st.expander("📉 Daily PL distribution"):
        import plotly.graph_objects as go
        pl = result.daily_pl
        fig_h = go.Figure(go.Histogram(x=pl, nbinsx=60, marker_color="#4269d0"))
        fig_h.add_vline(x=0, line_color="#888")
        fig_h.add_vline(x=float(np.mean(pl)), line_color="#c62828", line_dash="dash",
                        annotation_text=f"mean {np.mean(pl):.2f}")
        fig_h.update_layout(template="plotly_white", height=350,
                            xaxis_title="daily PL ($)", yaxis_title="days")
        st.plotly_chart(fig_h, width="stretch")


if __name__ == "__main__":
    main()
