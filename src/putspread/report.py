"""Slice 5 -- a single self-contained HTML report per run (STRATEGY.md section 9).

Emphasis, per the build prompt, on the two things that reveal whether the edge is
real: the P&L distribution and the fill-sensitivity comparison. Data caveats are
rendered at the TOP in a banner, not in a footnote, because a result whose caveats
you had to scroll to find is a result that will be quoted without them.
"""

from __future__ import annotations

import base64
import html
import io
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .engine import BacktestResult
from .metrics import Metrics, compute_metrics, exit_reason_breakdown

_CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
     margin:0;padding:2rem;background:#fbfbfd;color:#1a1a1f;line-height:1.5}
h1{margin:0 0 .25rem}h2{margin-top:2.5rem;border-bottom:1px solid #e3e3ea;padding-bottom:.4rem}
.sub{color:#6a6a78;margin-bottom:1.5rem}
.caveat{background:#fff4e5;border-left:4px solid #d97706;padding:1rem 1.25rem;margin:1rem 0;border-radius:4px}
.caveat h3{margin:0 0 .5rem;font-size:1rem;color:#92400e}
.caveat ul{margin:0;padding-left:1.25rem}
table{border-collapse:collapse;width:100%;font-size:.9rem;margin:1rem 0}
th,td{padding:.45rem .7rem;text-align:right;border-bottom:1px solid #ececf2}
th{background:#f2f2f7;text-align:right;font-weight:600}
td:first-child,th:first-child{text-align:left}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:.75rem;margin:1rem 0}
.card{background:#fff;border:1px solid #e6e6ee;border-radius:6px;padding:.85rem 1rem}
.card .k{font-size:.75rem;color:#6a6a78;text-transform:uppercase;letter-spacing:.03em}
.card .v{font-size:1.35rem;font-weight:600;margin-top:.2rem}
.pos{color:#047857}.neg{color:#b91c1c}
img{max-width:100%;border:1px solid #e6e6ee;border-radius:6px;background:#fff}
.scroll{overflow-x:auto}
code{background:#f2f2f7;padding:.1rem .35rem;border-radius:3px;font-size:.85em}
"""


def _fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _equity_plot(equity: pd.DataFrame, starting: float) -> str:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    eq = equity["equity"]
    ax1.plot(eq.index, eq.values, lw=1.4, color="#2563eb")
    ax1.axhline(starting, color="#9ca3af", lw=0.8, ls="--")
    ax1.set_ylabel("Equity ($)")
    ax1.grid(alpha=0.25)
    dd = eq - eq.cummax()
    ax2.fill_between(dd.index, dd.values, 0, color="#dc2626", alpha=0.35)
    ax2.set_ylabel("Drawdown ($)")
    ax2.grid(alpha=0.25)
    fig.autofmt_xdate()
    return _fig_to_data_uri(fig)


def _histogram_plot(trades: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    pnl = trades["pnl"].to_numpy(dtype=float)
    ax.hist(pnl, bins=min(40, max(10, len(pnl) // 4)), color="#2563eb", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="#111", lw=1)
    ax.axvline(pnl.mean(), color="#dc2626", lw=1.4, ls="--", label=f"mean ${pnl.mean():,.0f}")
    ax.set_xlabel("Trade P&L ($)")
    ax.set_ylabel("Trades")
    ax.legend()
    ax.grid(alpha=0.25)
    return _fig_to_data_uri(fig)


def _cards(m: Metrics) -> str:
    def cell(k: str, v: str, cls: str = "") -> str:
        return f'<div class="card"><div class="k">{k}</div><div class="v {cls}">{v}</div></div>'

    sign = lambda x: "pos" if x > 0 else "neg" if x < 0 else ""
    return '<div class="grid">' + "".join([
        cell("Trades", f"{m.n_trades:,}"),
        cell("Win rate", f"{m.win_rate:.1%}"),
        cell("Total P&L", f"${m.total_pnl:,.0f}", sign(m.total_pnl)),
        cell("Total return", f"{m.total_return:.1%}", sign(m.total_return)),
        cell("CAGR", f"{m.cagr:.1%}", sign(m.cagr)),
        cell("Max drawdown", f"{m.max_drawdown_pct:.1%}", "neg"),
        cell("Sharpe", f"{m.sharpe:.2f}", sign(m.sharpe)),
        cell("Sortino", f"{m.sortino:.2f}", sign(m.sortino)),
        cell("Profit factor", f"{m.profit_factor:.2f}", sign(m.profit_factor - 1)),
        cell("Expectancy/trade", f"${m.expectancy_per_trade:,.0f}", sign(m.expectancy_per_trade)),
        cell("Avg win", f"${m.avg_win:,.0f}", "pos"),
        cell("Avg loss", f"${m.avg_loss:,.0f}", "neg"),
        cell("Worst trade", f"${m.worst_trade:,.0f}", "neg"),
        cell("P&L skew", f"{m.pnl_skew:.2f}"),
    ]) + "</div>"


def _table(df: pd.DataFrame, float_fmt: str = "{:,.2f}", max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "<p><em>none</em></p>"
    d = df if max_rows is None else df.head(max_rows)
    return '<div class="scroll">' + d.to_html(
        float_format=lambda x: float_fmt.format(x), border=0, na_rep="-"
    ) + "</div>"


def build_report(
    result: BacktestResult,
    out_path: str | Path,
    title: str = "Bull Put Spread Backtest",
    fill_sensitivity: pd.DataFrame | None = None,
    extra_sections: dict[str, str] | None = None,
) -> Path:
    """Render the run to a single self-contained HTML file. Returns the path."""
    trades = result.trade_frame
    m = compute_metrics(trades, result.equity_curve, result.cfg.starting_equity)

    caveats = list(result.data_caveats)
    if result.fills.model == "mid":
        caveats.insert(0, (
            "FILLS MODELED AT MID -- optimistic and not achievable. STRATEGY.md "
            "section 7 requires realistic fills for any headline result."
        ))
    caveat_html = ""
    if caveats:
        items = "".join(f"<li>{html.escape(c)}</li>" for c in caveats)
        caveat_html = f'<div class="caveat"><h3>Data and modeling caveats</h3><ul>{items}</ul></div>'

    params = pd.DataFrame(
        sorted(((k, str(v)) for k, v in asdict(result.cfg).items()), key=lambda kv: kv[0]),
        columns=["parameter", "value"],
    ).set_index("parameter")

    metric_tbl = pd.DataFrame(
        [(k, v) for k, v in m.to_dict().items()], columns=["metric", "value"]
    ).set_index("metric")

    parts = [
        f"<h1>{html.escape(title)}</h1>",
        f'<div class="sub">Generated {datetime.now():%Y-%m-%d %H:%M} &middot; '
        f'{", ".join(result.cfg.symbols)} &middot; {result.cfg.start} to {result.cfg.end} '
        f'&middot; fills: <code>{result.fills.model}</code></div>',
        caveat_html,
        _cards(m),
    ]

    if not result.equity_curve.empty:
        parts += ["<h2>Equity and drawdown</h2>",
                  f'<img src="{_equity_plot(result.equity_curve, result.cfg.starting_equity)}">']
    if not trades.empty:
        parts += [
            "<h2>P&amp;L distribution</h2>",
            "<p>Credit strategies are left-skewed by construction: many small wins, "
            "occasional large losses. The tail matters more than the mean.</p>",
            f'<img src="{_histogram_plot(trades)}">',
        ]
    if fill_sensitivity is not None and not fill_sensitivity.empty:
        parts += [
            "<h2>Fill sensitivity</h2>",
            "<p>Same signals, same path, different execution assumptions. If the edge "
            "only exists at mid, the edge does not exist.</p>",
            _table(fill_sensitivity),
        ]
    parts += ["<h2>Metrics</h2>", _table(metric_tbl)]
    breakdown = exit_reason_breakdown(trades)
    if not breakdown.empty:
        parts += ["<h2>Exits</h2>", _table(breakdown)]
    for name, html_block in (extra_sections or {}).items():
        parts += [f"<h2>{html.escape(name)}</h2>", html_block]
    parts += ["<h2>Trade log</h2>", _table(trades, max_rows=500)]
    if not result.rejections.empty:
        counts = (result.rejections.groupby("reason").size()
                  .sort_values(ascending=False).to_frame("count"))
        parts += ["<h2>Why signals did not become trades</h2>", _table(counts, "{:,.0f}")]
    parts += ["<h2>Parameters</h2>", _table(params)]

    doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
           f"<title>{html.escape(title)}</title><style>{_CSS}</style></head>"
           f"<body>{''.join(parts)}</body></html>")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    return out
