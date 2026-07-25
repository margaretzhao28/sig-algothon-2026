"""
PyQt6 + pyqtgraph desktop dashboard for backtest results.

    .venv/bin/python viz_qt.py             # opens the last backtest run
    .venv/bin/python viz_qt.py <run.pkl>   # opens a specific saved run

run_backtest.py launches this automatically after every run (--no-viz to skip).

Layout
------
  top bar       : score / mean PL / sharpe / total PL / max DD / hit rate + Reload
  center        : three stacked plots sharing one time axis —
                    1. prices (normalized) with buy ▲ / sell ▼ markers
                    2. cumulative PnL line + daily PnL bars
                    3. dollar position (step)
  right sidebar : tabs —
                    Symbols  : show/hide checkbox per instrument + Traded/All/None
                    Stats    : full metrics table + per-instrument breakdown
                    Settings : display options (normalize, markers, sizes, panels)

Leave the window open while iterating: hit "⟳ Reload" after each new
run_backtest.py to pick up results/last_run.pkl.
"""

from __future__ import annotations
import os
import pickle
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from backtest.visualize import load_instrument_names

DEFAULT_PKL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "last_run.pkl")

PALETTE = [
    "#4fc3f7", "#ffb74d", "#ff8a65", "#4db6ac", "#81c784",
    "#f06292", "#ba68c8", "#90caf9", "#a1887f", "#e0e0e0",
]

# fair-price overlays get their own colors so they never blend in with the
# instrument price lines drawn from PALETTE
TREND_PALETTE = [
    "#ffd60a", "#ff2d95", "#00e5ff", "#adff2f", "#ff6d00",
    "#d500f9", "#76ff03", "#ff1744",
]

pg.setConfigOptions(antialias=True)

STATS_ROWS = [
    ("Score (2026)", "score", "{:.4f}"),
    ("Mean daily PL", "mean_pl", "${:.2f}"),
    ("Std daily PL", "std_pl", "${:.2f}"),
    ("Ann. Sharpe", "ann_sharpe", "{:.3f}"),
    ("Sortino", "sortino", "{:.3f}"),
    ("Calmar", "calmar_ratio", "{:.3f}"),
    ("t-stat", "t_stat", "{:.2f}"),
    ("p-value", "p_value", "{:.3f}"),
    ("Hit rate", "hit_rate", "{:.1%}"),
    ("Profit factor", "profit_factor", "{:.2f}"),
    ("Max drawdown", "max_drawdown", "${:.0f}"),
    ("Max DD duration", "max_drawdown_duration_days", "{:.0f} d"),
    ("% time underwater", "pct_time_underwater", "{:.1%}"),
    ("VaR 95%", "var_95", "${:.0f}"),
    ("CVaR 95%", "cvar_95", "${:.0f}"),
    ("Best day", "best_day_pl", "${:.0f}"),
    ("Worst day", "worst_day_pl", "${:.0f}"),
    ("Skewness", "skewness", "{:.2f}"),
    ("Lag-1 autocorr", "autocorr_lag1", "{:.3f}"),
    ("Total $ volume", "total_dvolume", "${:,.0f}"),
    ("Total commission", "total_commission", "${:.2f}"),
    ("Avg daily turnover", "avg_daily_turnover_ratio", "{:.1%}"),
    ("Avg gross exposure", "gross_exposure_avg", "${:,.0f}"),
    ("Instruments traded", "n_instruments_traded", "{:.0f}"),
    ("Eval days", "n_eval_days", "{:.0f}"),
]


_RANK_RE = re.compile(r"(LONG|SHORT) #(\d+) of (\d+)")


def _rank_tag(note) -> str:
    """Compact ridge-rank tag for the hover title, from the pos_annot cell, e.g.
    '<br>ridge LONG #2 of 13  (fc +0.0031)' -> '  [L#2/13]'.  '' if none."""
    if not note:
        return ""
    m = _RANK_RE.search(note)
    return f"  [{m.group(1)[0]}#{m.group(2)}/{m.group(3)}]" if m else ""


def _color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def _trend_color(i: int) -> str:
    return TREND_PALETTE[i % len(TREND_PALETTE)]


# ---- Pairs in src/strategies/pairs_9_fixed_beta.py (strategy indices) ----
ENGINE_PAIRS = [(29, 37), (10, 46), (1, 20), (13, 45), (49, 50),
                (8, 27), (25, 29), (25, 46), (25, 37)]
ENGINE_MEANREV = set()
ENGINE_MOMENTUM = {}
_PARTNERS = {}
for _a, _b in ENGINE_PAIRS:
    _PARTNERS.setdefault(_a, []).append(_b)
    _PARTNERS.setdefault(_b, []).append(_a)


def _lag1_autocorr(series) -> float:
    """lag-1 autocorrelation of a 1-D return series; nan if degenerate.
    This is the MR engine's predictive-power metric -- more negative = the stock
    reverts its own last move more strongly."""
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 5 or s.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(s[:-1], s[1:])[0, 1])


def _engine_label(j, names, ac, sharpe, pl):
    """(list_text, tooltip) annotating instrument j with its engine + a fit metric.
      pair  -> names its partner
      MR    -> lag-1 autocorr (predictive power; negative is good)
      ridge -> realized per-instrument Sharpe (how well the ridge actually did)"""
    name = names[j]
    ac_s = "n/a" if ac != ac else f"{ac:+.2f}"                       # nan-safe
    sh_s = "n/a" if sharpe != sharpe else f"{sharpe:+.2f}"
    pl_s = "n/a" if pl != pl else f"${pl:,.0f}"
    perf = f"realized Sharpe {sh_s} · PL {pl_s}"
    if j == 0:
        return f"{name}  · index (untraded)", f"{name} — equal-weight index (ALGO), not traded"
    if j in ENGINE_MEANREV:
        return (f"{name}  · MR  ac={ac_s}",
                f"{name} — MEAN-REVERSION engine\n"
                f"predictive power: lag-1 autocorr = {ac_s}  (more negative ⇒ stronger MR edge)\n"
                f"{perf}")
    if j in ENGINE_MOMENTUM:
        W = ENGINE_MOMENTUM[j]
        return (f"{name}  · MOM  MA{W}d",
                f"{name} — MOMENTUM (slow trend-following) engine\n"
                f"long above / short below its {W}-day moving average (rides long trend regimes)\n"
                f"{perf}")
    if j in _PARTNERS:
        partners = ", ".join(names[k] for k in _PARTNERS[j])
        return (f"{name}  · pair w/ {partners}",
                f"{name} — PAIR (cointegration) traded against {partners}\n{perf}")
    return (f"{name}  · untraded",
            f"{name} — not used by the selected pairs strategy")


class Dashboard(QtWidgets.QMainWindow):
    def __init__(self, pkl_path: str):
        super().__init__()
        self.pkl_path = pkl_path
        self.setWindowTitle("Backtest Dashboard")
        self.resize(1600, 950)

        # ---- top bar ----
        self.tiles = {}
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(12, 8, 12, 8)
        self.run_label = QtWidgets.QLabel("")
        bar.addWidget(self.run_label)
        bar.addStretch()
        for key, label in [("score", "SCORE 2026"), ("mean", "MEAN PL/DAY"),
                           ("sharpe", "SHARPE"), ("total", "TOTAL PL"),
                           ("dd", "MAX DD"), ("hit", "HIT RATE")]:
            v = QtWidgets.QLabel(f"{label}: —")
            bar.addWidget(v)
            bar.addSpacing(12)
            self.tiles[key] = v
        reload_btn = QtWidgets.QPushButton("Reload")
        reload_btn.clicked.connect(self.reload)
        bar.addWidget(reload_btn)

        # ---- plots (center) ----
        self.glw = pg.GraphicsLayoutWidget()
        self.p_price = self.glw.addPlot(row=0, col=0)
        self.p_pnl = self.glw.addPlot(row=1, col=0)
        self.p_pos = self.glw.addPlot(row=2, col=0)
        self.glw.ci.layout.setRowStretchFactor(0, 5)
        self.glw.ci.layout.setRowStretchFactor(1, 3)
        self.glw.ci.layout.setRowStretchFactor(2, 2)
        self.p_pnl.setXLink(self.p_price)
        self.p_pos.setXLink(self.p_price)
        self.p_price.setLabel("left", "price (norm)")
        self.p_pnl.setLabel("left", "PnL ($)")
        self.p_pos.setLabel("left", "position ($)")
        self.p_pos.setLabel("bottom", "day")
        for p in (self.p_price, self.p_pnl, self.p_pos):
            p.showGrid(x=True, y=True, alpha=0.15)
        self.p_price.addLegend(offset=(10, 10), labelTextSize="8pt")

        # ---- hover crosshair: one vertical line per panel, moved together ----
        self.crosshairs = []
        for p in (self.p_price, self.p_pnl, self.p_pos):
            line = pg.InfiniteLine(
                angle=90, movable=False,
                pen=pg.mkPen("#8b949e", width=1,
                             style=QtCore.Qt.PenStyle.DashLine))
            line.setZValue(10)
            line.hide()
            self.crosshairs.append(line)
        self._clear_hover_titles()
        self.glw.scene().sigMouseMoved.connect(self._on_mouse_moved)

        # ---- right sidebar: tabs ----
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setMaximumWidth(300)
        self.tabs.setMinimumWidth(240)
        self.tabs.addTab(self._make_symbols_tab(), "Symbols")
        self.tabs.addTab(self._make_trends_tab(), "Trends")
        self.tabs.addTab(self._make_stats_tab(), "Stats")
        self.tabs.addTab(self._make_settings_tab(), "Settings")

        # ---- assemble ----
        central = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        barw = QtWidgets.QWidget()
        barw.setLayout(bar)
        v.addWidget(barw)
        h = QtWidgets.QHBoxLayout()
        h.setContentsMargins(6, 6, 6, 6)
        h.addWidget(self.glw, stretch=1)
        h.addWidget(self.tabs)
        v.addLayout(h)
        self.setCentralWidget(central)
        self.reload()

    # -------------------------------------------------------------- tabs
    def _make_symbols_tab(self) -> QtWidgets.QWidget:
        self.list = QtWidgets.QListWidget()
        self.list.itemChanged.connect(lambda _i: self.redraw())

        btns = QtWidgets.QHBoxLayout()
        for text, fn in [("Traded", self._check_traded), ("All", self._check_all),
                         ("None", self._check_none)]:
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            btns.addWidget(b)

        self.sym_filter = QtWidgets.QLineEdit()
        self.sym_filter.setPlaceholderText("filter…")
        self.sym_filter.textChanged.connect(self._filter_symbols)

        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.addLayout(btns)
        lay.addWidget(self.sym_filter)
        lay.addWidget(self.list)
        return w

    def _make_trends_tab(self) -> QtWidgets.QWidget:
        self.trend_note = QtWidgets.QLabel(
            "No fair-price data — strategy has no getFairPrice().")
        self.trend_note.setWordWrap(True)
        self.trend_note.setStyleSheet("color: #888;")

        self.trend_list = QtWidgets.QListWidget()
        self.trend_list.itemChanged.connect(lambda _i: self.redraw())

        self.chk_trend_dash = QtWidgets.QCheckBox("Dashed line style")
        self.chk_trend_dash.setChecked(True)
        self.chk_trend_dash.stateChanged.connect(self.redraw)

        self.chk_trend_norm = QtWidgets.QCheckBox(
            "Normalize price to trend line (% deviation from fair)")
        self.chk_trend_norm.setToolTip(
            "Divide each instrument's price by its first checked fair line and plot\n"
            "the % deviation — the fair line becomes the flat 0 line, so mean\n"
            "reversion shows as oscillation around 0.")
        self.chk_trend_norm.stateChanged.connect(self.redraw)

        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(QtWidgets.QLabel("Fair price overlay (from getFairPrice)"))
        lay.addWidget(self.trend_note)
        lay.addWidget(self.trend_list)
        lay.addWidget(self.chk_trend_dash)
        lay.addWidget(self.chk_trend_norm)
        return w

    def _make_stats_tab(self) -> QtWidgets.QWidget:
        self.stats_table = QtWidgets.QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.horizontalHeader().setVisible(False)
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        self.stats_table.setEditTriggers(QtWidgets.QTableWidget.EditTrigger.NoEditTriggers)
        self.stats_table.setShowGrid(False)

        self.inst_table = QtWidgets.QTableWidget()
        self.inst_table.setColumnCount(4)
        self.inst_table.setHorizontalHeaderLabels(["Sym", "PL $", "Shrp", "Trds"])
        self.inst_table.verticalHeader().setVisible(False)
        self.inst_table.horizontalHeader().setStretchLastSection(True)
        self.inst_table.setEditTriggers(QtWidgets.QTableWidget.EditTrigger.NoEditTriggers)
        self.inst_table.setShowGrid(False)

        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(QtWidgets.QLabel("Run metrics"))
        lay.addWidget(self.stats_table, stretch=3)
        lay.addWidget(QtWidgets.QLabel("Per instrument (traded)"))
        lay.addWidget(self.inst_table, stretch=2)
        return w

    def _make_settings_tab(self) -> QtWidgets.QWidget:
        self.chk_norm = QtWidgets.QCheckBox("Normalize prices (start = 100)")
        self.chk_norm.setChecked(False)
        self.chk_marks = QtWidgets.QCheckBox("Buy/sell markers")
        self.chk_marks.setChecked(True)
        self.chk_bars = QtWidgets.QCheckBox("Daily PnL bars")
        self.chk_bars.setChecked(True)
        self.chk_cum = QtWidgets.QCheckBox("Cumulative PnL line")
        self.chk_cum.setChecked(True)
        self.chk_fill = QtWidgets.QCheckBox("Fill position area")
        self.chk_fill.setChecked(True)
        self.chk_pos_shares = QtWidgets.QCheckBox("Plot positions in shares (not $)")
        self.chk_pos_shares.setChecked(False)

        self.chk_xlimit = QtWidgets.QCheckBox("Lock time axis to data (no scrolling past ends)")
        self.chk_xlimit.setChecked(True)
        self.chk_hscroll = QtWidgets.QCheckBox("Horizontal-only scroll/zoom (y auto-fits)")
        self.chk_hscroll.setChecked(True)

        self.marker_size = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.marker_size.setRange(5, 20)
        self.marker_size.setValue(11)
        self.line_width = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.line_width.setRange(1, 5)
        self.line_width.setValue(1)

        for wdg in (self.chk_norm, self.chk_marks, self.chk_bars,
                    self.chk_cum, self.chk_fill, self.chk_pos_shares):
            wdg.stateChanged.connect(self.redraw)
        self.chk_xlimit.stateChanged.connect(self._apply_view_settings)
        self.chk_hscroll.stateChanged.connect(self._apply_view_settings)
        self.marker_size.valueChanged.connect(self.redraw)
        self.line_width.valueChanged.connect(self.redraw)

        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(QtWidgets.QLabel("Price panel"))
        lay.addWidget(self.chk_norm)
        lay.addWidget(self.chk_marks)
        lay.addWidget(QtWidgets.QLabel("Marker size"))
        lay.addWidget(self.marker_size)
        lay.addWidget(QtWidgets.QLabel("Line width"))
        lay.addWidget(self.line_width)
        lay.addSpacing(10)
        lay.addWidget(QtWidgets.QLabel("PnL panel"))
        lay.addWidget(self.chk_bars)
        lay.addWidget(self.chk_cum)
        lay.addSpacing(10)
        lay.addWidget(QtWidgets.QLabel("Position panel"))
        lay.addWidget(self.chk_pos_shares)
        lay.addWidget(self.chk_fill)
        lay.addSpacing(10)
        lay.addWidget(QtWidgets.QLabel("Navigation"))
        lay.addWidget(self.chk_xlimit)
        lay.addWidget(self.chk_hscroll)
        lay.addStretch()
        return w

    # -------------------------------------------------------------- data
    def reload(self):
        with open(self.pkl_path, "rb") as f:
            self.run = pickle.load(f)
        r = self.run["result"]
        self.names = (load_instrument_names(self.run.get("prices_file", "prices.txt"))
                      or [f"Inst_{j:02d}" for j in range(r.nInst)])
        self.traded = set(np.where(np.abs(r.delta_positions).sum(axis=0) > 0)[0].tolist())

        m = self.run["metrics"]
        days = r.eval_days
        self.run_label.setText(
            f"{self.run['strategy']}  ·  {self.run['dataset']}  ·  "
            f"days {days[0]}–{days[-1]}  ·  {self.run['timestamp']}")

        labels = {"score": "SCORE 2026", "mean": "MEAN PL/DAY", "sharpe": "SHARPE",
                  "total": "TOTAL PL", "dd": "MAX DD", "hit": "HIT RATE"}
        vals = {"score": f"{m['score']:.3f}", "mean": f"${m['mean_pl']:.2f}",
                "sharpe": f"{m['ann_sharpe']:.2f}",
                "total": f"${r.cumulative_value[-1]:,.0f}",
                "dd": f"${abs(m['max_drawdown']):,.0f}",
                "hit": f"{m['hit_rate']*100:.0f}%"}
        for key in self.tiles:
            self.tiles[key].setText(f"{labels[key]}: {vals[key]}")

        # symbols tab -- annotate each with its engine + a fit metric
        px = r.prices
        if isinstance(px, np.ndarray) and px.ndim == 2 and px.shape[0] > 5:
            rr = px[1:] / px[:-1] - 1.0
            autocorr = np.array([_lag1_autocorr(rr[:, j]) for j in range(r.nInst)])
        else:
            autocorr = np.full(r.nInst, np.nan)
        isharpe = m["inst_sharpe"]
        ipl = m["inst_pl_total"]

        self.list.blockSignals(True)
        self.list.clear()
        for j, name in enumerate(self.names):
            text, tip = _engine_label(j, self.names, autocorr[j],
                                      float(isharpe[j]), float(ipl[j]))
            item = QtWidgets.QListWidgetItem(text)
            item.setToolTip(tip)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked if j in self.traded
                               else QtCore.Qt.CheckState.Unchecked)
            item.setForeground(QtGui.QColor(_color(j)))
            self.list.addItem(item)
        self.list.blockSignals(False)
        self._filter_symbols(self.sym_filter.text())

        # trends tab: one row per (series, instrument) with a getFairPrice value
        fp = r.fair_price
        if isinstance(fp, np.ndarray):     # pickle from the single-series format
            fp = {"fair": fp}
        self.fair_price = fp or {}
        self.trend_list.blockSignals(True)
        self.trend_list.clear()
        k = 0
        for key, arr in self.fair_price.items():
            for j in np.where(~np.isnan(arr).all(axis=0))[0]:
                item = QtWidgets.QListWidgetItem(f"{self.names[j]} · {key}")
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.CheckState.Unchecked)   # overlays off by default
                item.setForeground(QtGui.QColor(_trend_color(k)))
                item.setData(QtCore.Qt.ItemDataRole.UserRole, (key, int(j), k))
                self.trend_list.addItem(item)
                k += 1
        self.trend_note.setVisible(k == 0)
        self.trend_list.blockSignals(False)

        # stats tab
        self.stats_table.setRowCount(len(STATS_ROWS))
        for i, (label, key, fmt) in enumerate(STATS_ROWS):
            self.stats_table.setItem(i, 0, QtWidgets.QTableWidgetItem(label))
            try:
                text = fmt.format(m[key])
            except (KeyError, ValueError, TypeError):
                text = "—"
            it = QtWidgets.QTableWidgetItem(text)
            it.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight
                                | QtCore.Qt.AlignmentFlag.AlignVCenter)
            self.stats_table.setItem(i, 1, it)
        self.stats_table.resizeRowsToContents()

        ipl = m["inst_pl_total"]
        isharpe = m["inst_sharpe"]
        n_trades = (r.delta_positions != 0).sum(axis=0)
        rows = sorted(self.traded, key=lambda j: -ipl[j])
        self.inst_table.setRowCount(len(rows))
        for i, j in enumerate(rows):
            for col, text in enumerate([self.names[j], f"{ipl[j]:,.0f}",
                                        f"{isharpe[j]:.2f}", f"{n_trades[j]}"]):
                it = QtWidgets.QTableWidgetItem(text)
                if col == 0:
                    it.setForeground(QtGui.QColor(_color(j)))
                else:
                    it.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight
                                        | QtCore.Qt.AlignmentFlag.AlignVCenter)
                self.inst_table.setItem(i, col, it)
        self.inst_table.resizeColumnsToContents()

        self.redraw()

    # ----------------------------------------------------------- symbols
    def _filter_symbols(self, text: str):
        text = text.strip().upper()
        for j in range(self.list.count()):
            item = self.list.item(j)
            item.setHidden(bool(text) and text not in item.text().upper())

    def _checked(self) -> list[int]:
        return [j for j in range(self.list.count())
                if self.list.item(j).checkState() == QtCore.Qt.CheckState.Checked]

    def _set_all(self, indices: set[int]):
        self.list.blockSignals(True)
        for j in range(self.list.count()):
            self.list.item(j).setCheckState(
                QtCore.Qt.CheckState.Checked if j in indices
                else QtCore.Qt.CheckState.Unchecked)
        self.list.blockSignals(False)
        self.redraw()

    def _check_traded(self): self._set_all(self.traded)
    def _check_all(self): self._set_all(set(range(self.list.count())))
    def _check_none(self): self._set_all(set())

    # -------------------------------------------------------------- view
    def _apply_view_settings(self):
        if not hasattr(self, "run"):
            return
        days = self.run["result"].eval_days
        pad = max(2.0, (days[-1] - days[0]) * 0.02)
        for p in (self.p_price, self.p_pnl, self.p_pos):
            vb = p.getViewBox()
            if self.chk_xlimit.isChecked():
                vb.setLimits(xMin=days[0] - pad, xMax=days[-1] + pad)
            else:
                vb.setLimits(xMin=None, xMax=None)
            if self.chk_hscroll.isChecked():
                # wheel/drag only move the time axis; y rescales to visible data
                vb.setMouseEnabled(x=True, y=False)
                vb.setAutoVisible(y=True)
                vb.enableAutoRange(y=True)
            else:
                vb.setMouseEnabled(x=True, y=True)
                vb.setAutoVisible(y=False)

    # ------------------------------------------------------------- hover
    def _clear_hover_titles(self):
        # keep a blank title reserved so the layout doesn't jump when text appears
        for p in (self.p_price, self.p_pnl, self.p_pos):
            p.setTitle(" ", size="9pt")

    def _on_mouse_moved(self, pos):
        h = getattr(self, "_hover", None)
        if h is None:
            return
        if not any(p.sceneBoundingRect().contains(pos)
                   for p in (self.p_price, self.p_pnl, self.p_pos)):
            for line in self.crosshairs:
                line.hide()
            self._clear_hover_titles()
            return

        x = self.p_price.getViewBox().mapSceneToView(pos).x()
        days = h["days"]
        i = int(np.argmin(np.abs(days - x)))
        d = days[i]
        for line in self.crosshairs:
            line.setPos(d)
            line.show()

        def fmt(items, limit=5):
            extra = f"   +{len(items) - limit} more" if len(items) > limit else ""
            return "    ".join(items[:limit]) + extra

        annot = h.get("pos_annot")
        price_parts = [
            f"{self.names[j]} ${h['prices'][i, j]:,.2f}"
            f"{_rank_tag(annot[i, j]) if annot is not None else ''}"
            for j in h["checked"]
        ]
        self.p_price.setTitle(f"day {d}    {fmt(price_parts)}" if price_parts
                              else f"day {d}", size="9pt", color="#e6edf3")
        self.p_pnl.setTitle(
            f"day {d}    cum PnL ${h['cum'][i]:,.2f}    daily {h['pl'][i]:+,.2f}",
            size="9pt", color="#e6edf3")
        pos_parts = [f"{self.names[j]} {h['shares'][i, j]:.0f} sh"
                     f" (${h['dollar'][i, j]:,.0f})"
                     for j in h["checked"] if np.any(h["shares"][:, j])]
        self.p_pos.setTitle(f"day {d}    {fmt(pos_parts)}" if pos_parts else " ",
                            size="9pt", color="#e6edf3")

    # -------------------------------------------------------------- plot
    def redraw(self):
        r = self.run["result"]
        days = r.eval_days
        prices = r.prices_prev
        norm = self.chk_norm.isChecked()
        y_price = 100.0 * prices / prices[0] if norm else prices

        pos_shares = self.chk_pos_shares.isChecked()
        self.p_pos.setLabel("left",
                            "position (shares)" if pos_shares else "position ($)")

        # Detrend mode: divide price by the instrument's first checked fair line
        # and plot % deviation (fair line itself becomes the flat 0 line).
        detrend = self.chk_trend_norm.isChecked()
        trend_ref = {}   # instrument j -> fair-price key used as its reference
        for row in range(self.trend_list.count()):
            item = self.trend_list.item(row)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                key, j, _k = item.data(QtCore.Qt.ItemDataRole.UserRole)
                trend_ref.setdefault(j, key)
        if detrend and trend_ref:
            self.p_price.setLabel("left", "deviation from fair (%)")
        else:
            self.p_price.setLabel(
                "left", "price (norm, start=100)" if norm else "price ($)")

        def _y_series(j):
            """Y data for instrument j on the price panel (honours detrend mode)."""
            if detrend and j in trend_ref:
                return 100.0 * (prices[:, j] / self.fair_price[trend_ref[j]][:, j] - 1.0)
            return y_price[:, j]

        checked = self._checked()
        lw = self.line_width.value()
        msize = self.marker_size.value()

        # PnL of ONLY the selected instruments: exact per-instrument daily PL
        # (mark-to-market minus that trade's commission), summed over the checked
        # set. Row sums reconcile to r.daily_pl when every instrument is checked.
        inst_pl = (r.positions_held * (r.prices - r.prices_prev)
                   - r.daily_commission_inst)
        sel = sorted(checked)
        pl_sel = inst_pl[:, sel].sum(axis=1) if sel else np.zeros(len(days))
        cum_sel = np.cumsum(pl_sel)

        self.p_price.clear()
        self.p_pos.clear()
        self.p_pnl.clear()

        # data the mouse-move handler reads to fill the crosshair readouts
        self._hover = dict(days=days, prices=prices, checked=checked,
                           shares=r.positions_held,
                           dollar=r.positions_held * prices,
                           pl=pl_sel, cum=cum_sel,
                           pos_annot=self.run.get("pos_annot"))
        # clear() removed the crosshair lines — put them back
        for p, line in zip((self.p_price, self.p_pnl, self.p_pos),
                           self.crosshairs):
            p.addItem(line, ignoreBounds=True)

        for j in checked:
            name = self.names[j]
            yj = _y_series(j)
            self.p_price.plot(days, yj, connect="finite",
                              pen=pg.mkPen(_color(j), width=lw), name=name)

            if self.chk_marks.isChecked():
                finite = np.isfinite(yj)
                if pos_shares:
                    # shares mode: a marker is any change in the SHARE position
                    delta = r.delta_positions[:, j].astype(float)
                    tip_buy = lambda x, y, data, n=name: (
                        f"{n} BUY {data[0]:.0f} sh @ ${data[1]:.2f}"
                        f" = ${data[0] * data[1]:,.0f}\nday {x:.0f}")
                    tip_sell = lambda x, y, data, n=name: (
                        f"{n} SELL {data[0]:.0f} sh @ ${data[1]:.2f}"
                        f" = ${data[0] * data[1]:,.0f}\nday {x:.0f}")
                else:
                    # $ mode: a marker is a change in the DOLLAR position. Rebalancing
                    # shares just to hold a constant $ target as the price drifts is NOT
                    # a trade here -- a capped name's $ position stays within one share
                    # of its target, so its day-to-day |delta$| < one share and is
                    # suppressed; only a real re-allocation ($ step) shows a marker.
                    dpos = r.positions_held[:, j] * prices[:, j]
                    delta = np.concatenate(([dpos[0]], np.diff(dpos)))
                    delta = np.where(np.abs(delta) >= prices[:, j], delta, 0.0)
                    tip_buy = lambda x, y, data, n=name: (
                        f"{n} BUY ${data[0]:,.0f} of position\nday {x:.0f}")
                    tip_sell = lambda x, y, data, n=name: (
                        f"{n} SELL ${data[0]:,.0f} of position\nday {x:.0f}")
                buys = np.where((delta > 0) & finite)[0]
                sells = np.where((delta < 0) & finite)[0]
                if len(buys):
                    self.p_price.addItem(pg.ScatterPlotItem(
                        x=days[buys], y=yj[buys], symbol="t1", size=msize,
                        brush=pg.mkBrush("#2ea043"), pen=pg.mkPen("w", width=0.5),
                        hoverable=True, hoverSize=msize + 4,
                        data=list(zip(delta[buys], prices[buys, j])),
                        tip=tip_buy))
                if len(sells):
                    self.p_price.addItem(pg.ScatterPlotItem(
                        x=days[sells], y=yj[sells], symbol="t", size=msize,
                        brush=pg.mkBrush("#f85149"), pen=pg.mkPen("w", width=0.5),
                        hoverable=True, hoverSize=msize + 4,
                        data=list(zip(-delta[sells], prices[sells, j])),
                        tip=tip_sell))

            shares = r.positions_held[:, j]
            y_pos = shares if pos_shares else shares * prices[:, j]
            if np.any(y_pos):
                kw = {}
                if self.chk_fill.isChecked():
                    c = QtGui.QColor(_color(j))
                    kw = dict(fillLevel=0,
                              brush=pg.mkBrush(c.red(), c.green(), c.blue(), 40))
                self.p_pos.plot(days, y_pos, pen=pg.mkPen(_color(j), width=lw),
                                stepMode="right", **kw)

        # fair-price overlays (from getFairPrice, if the strategy provides one)
        dash = (QtCore.Qt.PenStyle.DashLine if self.chk_trend_dash.isChecked()
                else QtCore.Qt.PenStyle.SolidLine)
        for row in range(self.trend_list.count()):
            item = self.trend_list.item(row)
            if item.checkState() != QtCore.Qt.CheckState.Checked:
                continue
            key, j, k = item.data(QtCore.Qt.ItemDataRole.UserRole)
            fp_j = self.fair_price[key][:, j]
            valid = ~np.isnan(fp_j)
            if not np.any(valid):
                continue
            if detrend and j in trend_ref:
                ref = self.fair_price[trend_ref[j]][:, j]
                y_fair = 100.0 * (fp_j / ref - 1.0)   # reference line -> flat 0
                valid &= np.isfinite(y_fair)
            else:
                y_fair = 100.0 * fp_j / prices[0, j] if norm else fp_j
            self.p_price.plot(days[valid], y_fair[valid],
                              pen=pg.mkPen(_trend_color(k), width=lw, style=dash),
                              name=f"{self.names[j]} {key}")

        # PnL panel — only the selected instruments (pl_sel / cum_sel above)
        pl = pl_sel
        self.p_pnl.setLabel("left", "PnL ($, all)" if len(sel) == self.list.count()
                            else f"PnL ($, {len(sel)} sel)")
        if self.chk_bars.isChecked():
            self.p_pnl.addItem(pg.BarGraphItem(
                x=days, height=pl, width=0.8,
                brushes=[pg.mkBrush("#2ea043" if v >= 0 else "#f85149") for v in pl],
                pen=pg.mkPen(None)))
        if self.chk_cum.isChecked():
            self.p_pnl.plot(days, cum_sel, pen=pg.mkPen("#e6edf3", width=2.2))
        self.p_pnl.addLine(y=0, pen=pg.mkPen("#8b949e", width=0.6,
                                             style=QtCore.Qt.PenStyle.DashLine))
        self.p_pos.addLine(y=0, pen=pg.mkPen("#8b949e", width=0.6,
                                             style=QtCore.Qt.PenStyle.DashLine))

        self.p_price.autoRange()
        self.p_pnl.autoRange()
        self.p_pos.autoRange()
        self._apply_view_settings()


def main() -> None:
    pkl_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PKL
    if not os.path.exists(pkl_path):
        sys.exit(f"No saved run at {pkl_path} — run a backtest first.")
    app = QtWidgets.QApplication(sys.argv)
    win = Dashboard(pkl_path)
    win.show()
    # macOS: windows spawned from a terminal subprocess open behind other apps
    # unless explicitly raised and the app is activated.
    win.raise_()
    win.activateWindow()
    if sys.platform == "darwin":
        try:
            from AppKit import NSApplication
            NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        except ImportError:
            pass
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
