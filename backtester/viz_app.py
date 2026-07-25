"""
Native desktop visualizer app for backtest results.

    .venv/bin/python viz_app.py            # opens the last backtest run
    .venv/bin/python viz_app.py <run.pkl>  # opens a specific saved run

run_backtest.py launches this automatically after every run (--no-viz to skip).
It reads results/last_run.pkl, renders the interactive three-panel chart
(prices + trade markers / PnL / positions, shared time axis) and shows it in a
native window.  The "Reload latest" button re-reads the pickle, so you can
leave the window open, re-run backtests in the terminal, and just click reload.
"""

from __future__ import annotations
import os
import pickle
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.visualize import build_figure, load_instrument_names

DEFAULT_PKL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "last_run.pkl")

_PAGE = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>
  body {{ font-family: -apple-system, 'Segoe UI', sans-serif; margin: 0; background: #fafafa; }}
  .bar {{ display: flex; align-items: center; gap: 24px; padding: 10px 18px;
         background: #111; color: #eee; }}
  .bar .title {{ font-weight: 700; font-size: 15px; margin-right: auto; }}
  .m {{ text-align: center; }}
  .m .v {{ font-size: 16px; font-weight: 700; }}
  .m .l {{ font-size: 10px; color: #aaa; text-transform: uppercase; letter-spacing: .5px; }}
  .pos {{ color: #7ee787; }} .neg {{ color: #ff7b72; }}
  button {{ background: #2ea043; color: white; border: 0; padding: 7px 14px;
           border-radius: 6px; font-weight: 600; cursor: pointer; }}
  button:hover {{ background: #3fb950; }}
  #chart {{ height: calc(100vh - 58px); }}
</style></head>
<body>
  <div class="bar">
    <span class="title">{title}</span>
    {tiles}
    <button onclick="pywebview.api.reload()">&#8635; Reload latest</button>
  </div>
  <div id="chart">{chart}</div>
</body>
</html>"""


def _tile(label: str, value: str, good: bool | None = None) -> str:
    cls = "" if good is None else (" pos" if good else " neg")
    return f'<div class="m"><div class="v{cls}">{value}</div><div class="l">{label}</div></div>'


def build_html(pkl_path: str) -> str:
    with open(pkl_path, "rb") as f:
        run = pickle.load(f)
    result, m = run["result"], run["metrics"]
    names = load_instrument_names(run.get("prices_file", "prices.txt"))

    fig = build_figure(result, instrument_names=names, title="")
    fig.update_layout(height=None, autosize=True, margin=dict(t=30, r=140))
    chart = fig.to_html(full_html=False, include_plotlyjs=True,
                        default_height="100%", config={"displaylogo": False})

    days = result.eval_days
    title = (f"{run['strategy']} · {run['dataset']} · days {days[0]}–{days[-1]}"
             f" · {run['timestamp']}")
    tiles = "".join([
        _tile("Score", f"{m['score']:.3f}", m["score"] > 0),
        _tile("Mean PL/day", f"${m['mean_pl']:.2f}", m["mean_pl"] > 0),
        _tile("Sharpe", f"{m['ann_sharpe']:.2f}", m["ann_sharpe"] > 0),
        _tile("Total PL", f"${result.cumulative_value[-1]:,.0f}",
              result.cumulative_value[-1] > 0),
        _tile("Max DD", f"${abs(m['max_drawdown']):,.0f}", False),
        _tile("Hit rate", f"{m['hit_rate']*100:.0f}%"),
    ])
    return _PAGE.format(title=title, tiles=tiles, chart=chart)


class Api:
    def __init__(self, pkl_path: str):
        self.pkl_path = pkl_path
        self.window = None

    def reload(self):
        self.window.load_url("file://" + _write_page(self.pkl_path))


def _write_page(pkl_path: str) -> str:
    html = build_html(pkl_path)
    out = os.path.join(tempfile.gettempdir(), "kevin_bt_viz.html")
    with open(out, "w") as f:
        f.write(html)
    return out


def main() -> None:
    pkl_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PKL
    if not os.path.exists(pkl_path):
        sys.exit(f"No saved run at {pkl_path} — run a backtest first.")

    import webview
    api = Api(pkl_path)
    api.window = webview.create_window(
        "Backtest Visualizer",
        url="file://" + _write_page(pkl_path),
        js_api=api,
        width=1500, height=980,
    )
    webview.start()


if __name__ == "__main__":
    main()
