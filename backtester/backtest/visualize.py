"""
Interactive HTML visualizer for BacktestResult.

Builds a three-panel plotly figure with a SHARED time axis:

  1. Normalized prices (start = 100) with buy/sell markers on every trade
  2. Cumulative PnL line + daily PnL bars
  3. Dollar position per instrument

Instruments the strategy actually traded are visible by default; all others
sit in the legend as click-to-show. Clicking a ticker in the legend toggles
its price line, trade markers, and position trace together (they share a
legendgroup). Drag-zoom on any panel zooms all three.

Usage from code:
    from backtest.visualize import save_and_open
    save_and_open(result, "viz.html", instrument_names=names, title="anso_mm")

The CLI (run_backtest.py) calls this automatically after a run unless
--no-viz is passed.
"""

from __future__ import annotations
import os
import webbrowser
from typing import Optional, Sequence

import numpy as np

from .engine import BacktestResult

# distinct-ish colors recycled across instruments
_PALETTE = [
    "#4269d0", "#efb118", "#ff725c", "#6cc5b0", "#3ca951",
    "#ff8ab7", "#a463f2", "#97bbf5", "#9c6b4e", "#9498a0",
]


def _color(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


def build_figure(
    result: BacktestResult,
    instrument_names: Optional[Sequence[str]] = None,
    title: str = "Backtest",
    instruments: Optional[Sequence[int]] = None,
    show_markers: bool = True,
    normalize: bool = True,
    pos_annot: Optional[np.ndarray] = None,
):
    """
    instruments : indices to include in the figure. None = all instruments,
                  with only the traded ones visible (rest legend-toggleable).
                  An explicit list includes ONLY those, all visible.
    pos_annot   : optional (n, nInst) object/str array. When given, entry [i, j]
                  (a string, e.g. "<br>ridge LONG #2 of 13") is appended to the
                  position hover for instrument j on row i. Use "" for no note.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n = len(result.daily_pl)
    days = result.eval_days                     # trading day t for row i
    prices = result.prices_prev                 # (n, nInst) price the strategy traded at on day t
    pos = result.positions_held                 # (n, nInst) shares held from day t
    delta = result.delta_positions              # (n, nInst) shares traded on day t

    nInst = result.nInst
    if instrument_names is None:
        instrument_names = [f"Inst_{j:02d}" for j in range(nInst)]

    traded = np.where(np.abs(delta).sum(axis=0) > 0)[0]
    if instruments is None:
        included = list(range(nInst))
        default_visible = set(traded.tolist() if len(traded) else range(min(5, nInst)))
    else:
        included = list(instruments)
        default_visible = set(included)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.45, 0.30, 0.25], vertical_spacing=0.04,
        specs=[[{}], [{"secondary_y": True}], [{}]],
        subplot_titles=(
            "Prices (normalized, start = 100) — click tickers in legend to show/hide",
            "PnL",
            "Position ($)",
        ),
    )

    # ---- Row 1: prices + trade markers --------------------------------------
    norm = 100.0 * prices / prices[0] if normalize else prices

    for j in included:
        name = instrument_names[j]
        visible = True if j in default_visible else "legendonly"
        if pos_annot is not None:
            cdp = np.empty((len(days), 2), dtype=object)
            cdp[:, 0] = prices[:, j]
            cdp[:, 1] = pos_annot[:, j]
            price_cd = cdp
            price_ht = (f"{name}  $%{{customdata[0]:.2f}}  (norm %{{y:.2f}})"
                        f"%{{customdata[1]}}<extra></extra>")
        else:
            price_cd = prices[:, j]
            price_ht = f"{name}  $%{{customdata:.2f}}  (norm %{{y:.2f}})<extra></extra>"
        fig.add_trace(go.Scatter(
            x=days, y=norm[:, j], mode="lines", name=name,
            legendgroup=name, visible=visible,
            line=dict(width=1.4, color=_color(j)),
            customdata=price_cd,
            hovertemplate=price_ht,
        ), row=1, col=1)

        # buy / sell markers for instruments that trade
        buys = np.where(delta[:, j] > 0)[0] if show_markers else np.array([], dtype=int)
        sells = np.where(delta[:, j] < 0)[0] if show_markers else np.array([], dtype=int)
        if len(buys):
            trade_info = np.column_stack([
                delta[buys, j], prices[buys, j],
                delta[buys, j] * prices[buys, j],
            ])
            fig.add_trace(go.Scatter(
                x=days[buys], y=norm[buys, j], mode="markers",
                name=f"{name} buys", legendgroup=name, showlegend=False,
                visible=visible,
                marker=dict(symbol="triangle-up", size=9, color="#2e7d32",
                            line=dict(width=1, color="white")),
                customdata=trade_info,
                hovertemplate=(f"{name} BUY %{{customdata[0]:.0f}} sh"
                               f" @ $%{{customdata[1]:.2f}}"
                               f" = $%{{customdata[2]:,.0f}}<extra></extra>"),
            ), row=1, col=1)
        if len(sells):
            trade_info = np.column_stack([
                -delta[sells, j], prices[sells, j],
                -delta[sells, j] * prices[sells, j],
            ])
            fig.add_trace(go.Scatter(
                x=days[sells], y=norm[sells, j], mode="markers",
                name=f"{name} sells", legendgroup=name, showlegend=False,
                visible=visible,
                marker=dict(symbol="triangle-down", size=9, color="#c62828",
                            line=dict(width=1, color="white")),
                customdata=trade_info,
                hovertemplate=(f"{name} SELL %{{customdata[0]:.0f}} sh"
                               f" @ $%{{customdata[1]:.2f}}"
                               f" = $%{{customdata[2]:,.0f}}<extra></extra>"),
            ), row=1, col=1)

    # ---- Row 2: cumulative PnL + daily PnL bars ----------------------------
    fig.add_trace(go.Scatter(
        x=days, y=result.cumulative_value, mode="lines", name="Cumulative PnL",
        line=dict(width=2.2, color="#111111"),
        hovertemplate="cum PnL $%{y:.2f}<extra></extra>",
    ), row=2, col=1, secondary_y=False)

    bar_colors = np.where(result.daily_pl >= 0, "#2e7d32", "#c62828")
    fig.add_trace(go.Bar(
        x=days, y=result.daily_pl, name="Daily PnL",
        marker_color=bar_colors, opacity=0.45,
        hovertemplate="daily PnL $%{y:.2f}<extra></extra>",
    ), row=2, col=1, secondary_y=True)

    # ---- Row 3: dollar position per instrument -----------------------------
    dollar_pos = pos * prices
    for j in included:
        if not np.any(pos[:, j]):
            continue
        name = instrument_names[j]
        visible = True if j in default_visible else "legendonly"
        fig.add_trace(go.Scatter(
            x=days, y=dollar_pos[:, j], mode="lines", name=f"{name} pos",
            legendgroup=name, showlegend=False, visible=visible,
            line=dict(width=1.4, color=_color(j), shape="hv"),
            fill="tozeroy",
            customdata=pos[:, j],
            hovertemplate=f"{name} pos %{{customdata:.0f}} sh = $%{{y:,.0f}}<extra></extra>",
        ), row=3, col=1)
    fig.add_hline(y=0, line_width=0.8, line_color="#888", row=3, col=1)

    # Bind every trace to the bottom x-axis so hovering anywhere produces ONE
    # unified box (price + trades + PnL + positions for that day) and the spike
    # line below spans all three panels as a single crosshair.
    fig.update_traces(xaxis="x3")

    fig.update_layout(
        title=title,
        height=950,
        template="plotly_white",
        hovermode="x unified",
        hoverlabel=dict(font=dict(size=11)),
        spikedistance=-1,
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.01,
                    font=dict(size=10)),
        margin=dict(r=140, t=80),
        bargap=0,
    )
    fig.update_xaxes(
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikecolor="#555", spikethickness=1, spikedash="dot",
        row=3, col=1,
    )
    fig.update_yaxes(title_text="norm price" if normalize else "price", row=1, col=1)
    fig.update_yaxes(title_text="cum PnL ($)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="daily PnL ($)", row=2, col=1, secondary_y=True,
                     showgrid=False)
    fig.update_yaxes(title_text="position ($)", row=3, col=1)
    fig.update_xaxes(title_text="day", row=3, col=1)
    return fig


def ridge_rank_annotations(prices, eval_days, strategy_module):
    """Build an (n, nInst) object array of ridge-tail hover notes for a strategy
    that exposes its ridge internals (_forecast_ridge_target, RIDGE_TARGET_INDICES,
    RIDGE_TAIL_SIZE, RIDGE_MIN_TRAINING_OBSERVATIONS; MEANREV_INDICES optional).

    Entry [i, j] is a string such as "<br>ridge LONG #2 of 13  (fc +0.0031)" telling
    where stock j's forecast ranked in the tail on trading day eval_days[i] -- #1 is
    the strongest long/short, #k the marginal one. "" where j isn't ridge-traded that
    day. Returns None if the strategy has no ridge to rank.

    The ranking is replayed with the EXACT history slice the engine feeds the
    strategy (prices[:, :t]) and mirrors _build_ridge_positions: tail size is
    min(RIDGE_TAIL_SIZE, n_forecasts // 2), and MEANREV stocks are ranked but skipped
    (another engine owns them, so they get no ridge note)."""
    m = strategy_module
    need = ("_forecast_ridge_target", "RIDGE_TARGET_INDICES", "RIDGE_TAIL_SIZE",
            "RIDGE_MIN_TRAINING_OBSERVATIONS")
    if not all(hasattr(m, a) for a in need):
        return None

    prices = np.asarray(prices, dtype=float)
    eval_days = np.asarray(eval_days, dtype=int)
    meanrev = {int(x) for x in getattr(m, "MEANREV_INDICES", ())}
    targets = [int(x) for x in m.RIDGE_TARGET_INDICES]
    min_obs = int(m.RIDGE_MIN_TRAINING_OBSERVATIONS)
    tail_cap = int(m.RIDGE_TAIL_SIZE)

    annot = np.full((len(eval_days), prices.shape[0]), "", dtype=object)
    for i, t in enumerate(eval_days):
        hist = prices[:, :t]
        if hist.shape[1] < min_obs + 2:
            continue
        all_returns = hist[:, 1:] / hist[:, :-1] - 1.0
        forecasts = {}
        for tg in targets:
            f = m._forecast_ridge_target(all_returns, tg)
            if f is not None and np.isfinite(f):
                forecasts[tg] = f
        n_fc = len(forecasts)
        tail = min(tail_cap, n_fc // 2)
        if tail < 1:
            continue
        ranked = sorted(forecasts, key=forecasts.get)        # ascending: [0]=most short
        longs = set(ranked[-tail:])
        shorts = set(ranked[:tail])
        for p, tg in enumerate(ranked):
            if tg in meanrev:
                continue
            if tg in longs:
                annot[i, tg] = (f"<br>ridge LONG #{n_fc - p} of {tail}"
                                f"  (fc {forecasts[tg]:+.4f})")
            elif tg in shorts:
                annot[i, tg] = (f"<br>ridge SHORT #{p + 1} of {tail}"
                                f"  (fc {forecasts[tg]:+.4f})")
    return annot


def load_instrument_names(prices_file: str) -> Optional[list[str]]:
    """Read tickers.txt next to the prices file (symlinks resolved), if present."""
    path = os.path.join(os.path.dirname(os.path.realpath(prices_file)), "tickers.txt")
    if os.path.exists(path):
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]
    return None


def save_and_open(
    result: BacktestResult,
    out_path: str,
    instrument_names: Optional[Sequence[str]] = None,
    title: str = "Backtest",
    open_browser: bool = True,
    pos_annot: Optional[np.ndarray] = None,
) -> str:
    """Write the interactive HTML report and (optionally) open it."""
    fig = build_figure(result, instrument_names=instrument_names, title=title,
                       pos_annot=pos_annot)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.write_html(out_path, include_plotlyjs=True)
    if open_browser:
        webbrowser.open("file://" + os.path.abspath(out_path))
    return out_path
