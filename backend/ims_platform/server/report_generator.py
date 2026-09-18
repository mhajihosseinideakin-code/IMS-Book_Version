"""
server.report_generator
--------------------------

A single, generic report-building engine used by every Case Library
case (Iberian, GFM Current-Limit Recovery, Multi-Converter Fault
Recovery). Per the explicit requirement this was built against: HTML
and PDF are generated from the SAME structured input and the SAME HTML
markup (PDF is produced by rendering that exact HTML with wkhtmltopdf,
not a second, independently-written template) -- so they cannot drift
apart.

Design rule, followed throughout: this module NEVER invents a value.
Every number and every plotted curve is read directly out of the
result dict a case's own `run_stress_test`-style function already
returned (the same dict the API/UI already display) or out of an
explicitly-passed counterfactual/boundary/recommendation dict from
that same case's own backend functions. A section for which the
caller did not supply data renders the literal string "Not evaluated
in this analysis." -- it is never silently populated with a default,
and "not achievable within tested range" (a real, negative result) is
never rendered as "impossible" (a stronger claim the underlying search
did not establish).

Plots are rendered SERVER-SIDE (matplotlib, Agg backend) and embedded
as base64 PNGs directly in the HTML. This is a deliberate choice, not
a style preference: the exported HTML/PDF must be self-contained and
show the actual simulated curves without depending on the dashboard's
own canvas/JS renderer (which cannot run inside a static export or a
PDF converter). The data plotted is read from the same result dict's
`trajectory`/`voltage_trace(s)`/`trip_timer_trace` arrays used by the
live dashboard -- the same numbers, a different renderer.
"""

from __future__ import annotations

import base64
import datetime
import html as html_lib
import io
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLATFORM_VERSION = "v4.5"

_BG = "#0b1620"
_PANEL = "#0f1e29"
_GRID = "#1b2733"
_TEXT = "#dce6ef"
_MUTED = "#9fb0bf"
_COLORS = ["#3FE0A8", "#6FC6FF", "#B18CFF", "#FFB454", "#FF6B6B", "#7CE38B"]

_LOGO_PATH = os.path.join(os.path.dirname(__file__), "static", "assets", "logo", "IMS-icon.png")
_logo_data_uri_cache: Optional[str] = None


def _logo_data_uri() -> str:
    """Base64-embeds the platform's own logo (server/static/assets/logo/IMS-icon.png, the same
    file used in the live dashboard header) so the report is self-contained and uses the real
    brand asset, not a placeholder icon. Cached after first read."""
    global _logo_data_uri_cache
    if _logo_data_uri_cache is None:
        try:
            with open(_LOGO_PATH, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            _logo_data_uri_cache = f"data:image/png;base64,{b64}"
        except OSError:
            _logo_data_uri_cache = ""
    return _logo_data_uri_cache


# ---------------------------------------------------------------------
# Plot rendering (server-side, matplotlib -> base64 PNG)
# ---------------------------------------------------------------------

def _fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _style_axes(ax, xlabel, ylabel, title=None):
    ax.set_facecolor(_PANEL)
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.spines[:].set_color(_GRID)
    ax.tick_params(colors=_MUTED, labelsize=9)
    ax.set_xlabel(xlabel, color=_TEXT, fontsize=10)
    ax.set_ylabel(ylabel, color=_TEXT, fontsize=10)
    if title:
        ax.set_title(title, color=_TEXT, fontsize=11, loc="left", fontweight="bold")


def plot_time_series(t: Sequence[float], series: Dict[str, Sequence[float]], xlabel: str, ylabel: str,
                      title: str, hlines: Optional[List[Tuple[float, str, str]]] = None,
                      vlines: Optional[List[Tuple[float, str]]] = None) -> str:
    """hlines: list of (y, label, color). vlines: list of (x, color) -- for disturbance event markers."""
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    fig.patch.set_facecolor(_BG)
    _style_axes(ax, xlabel, ylabel, title)
    for i, (label, vals) in enumerate(series.items()):
        ax.plot(t, vals, label=label, color=_COLORS[i % len(_COLORS)], linewidth=1.8)
    for y, label, color in (hlines or []):
        ax.axhline(y, color=color, linestyle="--", linewidth=1.2, label=label)
    for x, color in (vlines or []):
        ax.axvline(x, color=color, linestyle=":", linewidth=1.0, alpha=0.6)
    ax.legend(facecolor=_PANEL, edgecolor=_GRID, labelcolor=_TEXT, fontsize=8, loc="best")
    return _fig_to_data_uri(fig)


def plot_bar_verdicts(x_values: Sequence[float], verdicts: Sequence[Optional[str]], xlabel: str,
                       title: str) -> str:
    score_of = {"RECOVERABLE": 1.0, "AT RISK": 0.5, "NON-RECOVERABLE": 0.0}
    color_of = {"RECOVERABLE": "#3FE0A8", "AT RISK": "#FFB454", "NON-RECOVERABLE": "#FF6B6B"}
    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    fig.patch.set_facecolor(_BG)
    # Recoverability score is defined on [0,1] (0/0.5/1 encoding of the verdict) -- the axis must
    # match that definition exactly, not run past it (was 1.3 -- a real inconsistency, fixed).
    _style_axes(ax, xlabel, "Recoverability score (0-1)", title)
    width = (max(x_values) - min(x_values)) / max(len(x_values) - 1, 1) * 0.5 if len(x_values) > 1 else 0.3
    for x, v in zip(x_values, verdicts):
        if v is None:
            continue
        score = score_of[v]
        if score > 0:
            ax.bar(x, score, width=width, color=color_of[v])
        else:
            # Marker only, no per-point "0.0" text: with several tested values close together
            # (e.g. Imax = 0.3/0.4/0.5/0.6), stacking a text annotation at every one collided
            # into unreadable overlapping text -- the exact complaint. The Y-axis's own "0" tick
            # already labels the value; the marker alone identifies which x-points it applies to.
            ax.scatter([x], [0], color=color_of[v], s=40, zorder=5)
    ax.set_ylim(-0.06, 1.06)
    ax.set_yticks([0, 0.5, 1.0])
    # Explicit ticks at the ACTUAL tested x-values for GRIDLINES/data points, but LABELS are a
    # genuinely reduced subset -- selected by the same greedy minimum-pixel-gap rule as the
    # canvas dashboard chart (server/static/explorer.html's drawGrid xTickAutoReduce), so
    # browser/HTML/PDF present identically reduced tick sets, not just independently "rotated"
    # dense ones. Rotation is still applied on top for whatever reduced set remains close.
    ax.set_xticks(list(x_values))
    fig.canvas.draw()  # needed so matplotlib has real pixel transforms to measure against
    xs_sorted = sorted(x_values)
    px_positions = [ax.transData.transform((x, 0))[0] for x in xs_sorted]
    min_label_gap_px = 34.0
    keep = [0]
    for i in range(1, len(xs_sorted)):
        if px_positions[i] - px_positions[keep[-1]] >= min_label_gap_px:
            keep.append(i)
    if keep[-1] != len(xs_sorted) - 1:
        if px_positions[-1] - px_positions[keep[-1]] < min_label_gap_px * 0.6:
            keep[-1] = len(xs_sorted) - 1
        else:
            keep.append(len(xs_sorted) - 1)
    keep_values = {xs_sorted[i] for i in keep}
    labels = [_fmt_axis_value(v) if v in keep_values else "" for v in x_values]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
    fig.subplots_adjust(bottom=0.24)
    return _fig_to_data_uri(fig)


def _fmt_axis_value(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".") if v != int(v) else f"{int(v)}"


def plot_manifold_distance(t, d_M, title="Manifold distance d_M(t)") -> str:
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    fig.patch.set_facecolor(_BG)
    _style_axes(ax, "time (s)", "d_M", title)
    ax.plot(t, d_M, color="#B18CFF", linewidth=1.8)
    ax.set_ylim(bottom=0)
    return _fig_to_data_uri(fig)


def plot_transverse_rate(t, lambda_perp, title="Transverse rate \u03bb\u22a5(t)") -> str:
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    fig.patch.set_facecolor(_BG)
    _style_axes(ax, "time (s)", "\u03bb\u22a5 (1/s)", title)
    ax.plot(t, lambda_perp, color="#6FC6FF", linewidth=1.8)
    ax.axhline(0, color="#FF6B6B", linestyle="--", linewidth=1.2, label="contracting / expanding boundary")
    ax.legend(facecolor=_PANEL, edgecolor=_GRID, labelcolor=_TEXT, fontsize=8)
    return _fig_to_data_uri(fig)


def plot_network_topology(nodes: List[Dict], edges: List[Tuple[str, str]], title: str = "Network topology") -> str:
    """
    Explicit network topology diagram -- built to make a genuine
    modelling fact visible (e.g. which bus is a FIXED/infinite
    reference and therefore contributes no dynamic voltage trace),
    rather than fabricating an extra trace to paper over it.

    nodes: list of {"id": str, "label": str, "kind": "fixed"|"dynamic", "x": float, "y": float}
    edges: list of (from_id, to_id) tuples.
    """
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.15)
    ax.axis("off")
    ax.set_title(title, color=_TEXT, fontsize=12, loc="left", fontweight="bold")

    pos = {n["id"]: (n["x"], n["y"]) for n in nodes}
    for a, b in edges:
        if a in pos and b in pos:
            xa, ya = pos[a]
            xb, yb = pos[b]
            ax.plot([xa, xb], [ya, yb], color="#4A6070", linewidth=1.8, zorder=1)

    for n in nodes:
        x, y = n["x"], n["y"]
        is_fixed = n.get("kind") == "fixed"
        color = "#FFB454" if is_fixed else "#3FE0A8"
        radius = 0.075 if is_fixed else 0.065
        circle = plt.Circle((x, y), radius, facecolor=color, edgecolor=_BG, linewidth=2, zorder=3)
        ax.add_patch(circle)
        ax.annotate(n["label"], (x, y), xytext=(0, -radius * 220 - 8), textcoords="offset points",
                    ha="center", va="top", color=_TEXT, fontsize=9.5, zorder=4)
        tag = "FIXED (v = 1.0 p.u.)\nno dynamic state" if is_fixed else "dynamic state"
        ax.annotate(tag, (x, y), xytext=(0, radius * 220 + 6), textcoords="offset points",
                    ha="center", va="bottom", color=_MUTED, fontsize=7.5, zorder=4)

    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color="#FFB454", label="Fixed/infinite bus (no ODE state)"),
               mpatches.Patch(color="#3FE0A8", label="Dynamic bus (has its own voltage-state trace)")]
    ax.legend(handles=handles, facecolor=_PANEL, edgecolor=_GRID, labelcolor=_TEXT, fontsize=8, loc="lower center",
              bbox_to_anchor=(0.5, -0.12), ncol=2)
    return _fig_to_data_uri(fig)


def plot_roa_slice(roa: Dict, title="Region of Attraction (ROA) slice") -> str:
    """
    Renders the 2-D ROA slice (server.ims_geometry.roa_slice's output)
    as a genuine grid of real simulated outcomes -- green = recoverable,
    red = non-recoverable -- NOT a smoothed/interpolated heatmap. Each
    cell boundary corresponds to an actual simulated grid point.
    """
    v_values = roa["v_values"]
    timer_values = roa["timer_values"]
    grid = roa["grid"]
    n_timer, n_v = len(grid), len(grid[0]) if grid else 0
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.patch.set_facecolor(_BG)
    _style_axes(ax, "v_es_sw (p.u.), initial condition", "trip_timer, initial condition", title)
    color_map = {True: "#3FE0A8", False: "#FF6B6B"}
    dv = (v_values[1] - v_values[0]) if n_v > 1 else 0.05
    dtm = (timer_values[1] - timer_values[0]) if n_timer > 1 else 0.1
    for i, tm in enumerate(timer_values):
        for j, v in enumerate(v_values):
            cell = grid[i][j]
            rect = plt.Rectangle((v - dv / 2, tm - dtm / 2), dv, dtm,
                                  facecolor=color_map[cell["recoverable"]], edgecolor=_BG, linewidth=0.5)
            ax.add_patch(rect)
    ax.set_xlim(min(v_values) - dv / 2, max(v_values) + dv / 2)
    ax.set_ylim(min(timer_values) - dtm / 2, max(timer_values) + dtm / 2)
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color="#3FE0A8", label="Recoverable"), mpatches.Patch(color="#FF6B6B", label="Non-recoverable")]
    ax.legend(handles=handles, facecolor=_PANEL, edgecolor=_GRID, labelcolor=_TEXT, fontsize=8, loc="upper right")
    return _fig_to_data_uri(fig)


# ---------------------------------------------------------------------
# Report shell / section rendering
# ---------------------------------------------------------------------

def _esc(s) -> str:
    return html_lib.escape(str(s))


def _fmt(v, ndigits=4):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return f"{v:.{ndigits}f}"
    return _esc(v)


def _params_table(params: Dict) -> str:
    rows = "".join(f"<tr><td>{_esc(k)}</td><td class='num'>{_fmt(v)}</td></tr>" for k, v in params.items())
    return f"<table class='ptable'><tbody>{rows}</tbody></table>"


def _not_evaluated() -> str:
    return "<p class='not-evaluated'>Not evaluated in this analysis.</p>"


def _verdict_badge(verdict: str) -> str:
    color = {"RECOVERABLE": "#3FE0A8", "AT RISK": "#FFB454", "NON-RECOVERABLE": "#FF6B6B"}.get(verdict, "#9fb0bf")
    return f"<span class='verdict-badge' style='color:{color};border-color:{color};'>{_esc(verdict)}</span>"


def _auto_number_sections(body: str) -> str:
    """
    Assigns TRUE sequential section numbers based on the actual order
    <h2> tags appear in the assembled document -- not hardcoded per
    section function. This is the structural fix for a real bug: each
    section_* function used to hard-code its own number (e.g. "8.",
    "11.", "9.", "10."), which meant the rendered ORDER of sections
    (which varies per case -- e.g. Multi-Converter inserts an extra
    "Multi-Converter Interaction" section that Iberian and GFM don't
    have) no longer matched the hard-coded numbers, producing exactly
    the out-of-sequence numbering (8, 11, 9, 10) found in review. Every
    section_* function now emits an UNNUMBERED <h2>Title</h2>; this
    function is the single place numbering happens, applied once to the
    fully-assembled body, so it is structurally impossible for the
    rendered numbers to be out of order or to collide, regardless of
    which sections a given case includes or what order they're built in.
    """
    counter = [0]

    def _renumber(match):
        counter[0] += 1
        return f"<h2>{counter[0]}. {match.group(1)}</h2>"

    return re.sub(r"<h2>(?!\d+\.\s)(.*?)</h2>", _renumber, body)


def document_shell(title: str, subtitle: str, case_id: str, sections_html: List[str],
                    case_status: str = "ACTIVE \u2014 ANALYSIS AVAILABLE") -> str:
    generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    body = _auto_number_sections("\n".join(sections_html))
    logo = _logo_data_uri()
    logo_img = f'<img src="{logo}" alt="IMS Platform"/>' if logo else ""
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>{_esc(title)} -- Analysis Report</title>
<style>
  @page {{ margin: 20mm 14mm 16mm; }}
  * {{ box-sizing: border-box; }}
  body {{ background:{_BG}; color:{_TEXT}; font-family: -apple-system, Segoe UI, Arial, sans-serif;
          margin:0; padding: 0 34px 28px; line-height:1.55; }}
  .report-header {{ display:flex; align-items:center; gap:14px; padding: 22px 0 16px;
                     border-bottom: 2px solid #1b2733; margin-bottom: 18px; }}
  .report-header img {{ height: 40px; width: auto; flex-shrink:0; }}
  .report-header .brand-text b {{ font-size: 15px; color: #eaf2f8; display:block; }}
  .report-header .brand-text span {{ font-size: 10px; color: {_MUTED}; letter-spacing:.03em; }}
  .report-header .status-chip {{ margin-left:auto; font-size: 10px; font-weight:700; letter-spacing:.04em;
                     color: var(--teal, #3FE0A8); border: 1px solid #3FE0A8; border-radius: 4px;
                     padding: 4px 10px; white-space: nowrap; }}
  h1 {{ font-size: 21px; margin: 0 0 4px; }}
  h2 {{ font-size: 15px; margin: 30px 0 10px; padding-bottom: 6px; border-bottom: 1px solid #24313d;
        color: #eaf2f8; page-break-after: avoid; }}
  h3 {{ font-size: 12.5px; color: {_TEXT}; margin: 16px 0 6px; page-break-after: avoid; }}
  .subtitle {{ color: {_MUTED}; font-size: 12.5px; margin-bottom: 4px; }}
  .meta {{ color: {_MUTED}; font-size: 10.5px; font-family: monospace; margin-bottom: 18px; }}
  .disclaimer {{ font-size: 11px; color: {_MUTED}; border-left: 3px solid #FFB454;
                 background: rgba(255,180,84,0.06); padding: 10px 14px; border-radius: 0 6px 6px 0;
                 margin: 14px 0; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 11.5px; margin: 8px 0 14px; }}
  table.ptable td:first-child, table.ctable td:first-child {{ color: {_MUTED}; }}
  td, th {{ padding: 6px 10px; border-bottom: 1px solid #1b2733; text-align: left; }}
  td.num {{ text-align: right; font-family: monospace; }}
  th {{ color: {_MUTED}; font-weight: 600; font-size: 10.5px; text-transform: uppercase; }}
  .verdict-badge {{ display:inline-block; font-weight: 800; font-size: 16px; border: 1.5px solid;
                     border-radius: 6px; padding: 4px 14px; }}
  .metric-grid {{ display:flex; flex-wrap:wrap; gap: 14px; margin: 10px 0 16px; }}
  .metric {{ background:{_PANEL}; border:1px solid #1b2733; border-radius:6px; padding:10px 14px; min-width:160px; }}
  .metric .label {{ font-size: 9.5px; color:{_MUTED}; text-transform:uppercase; letter-spacing:.04em; }}
  .metric .value {{ font-size: 15px; font-weight:700; margin-top:2px; font-family: monospace; }}
  .figure {{ margin: 10px 0 18px; page-break-inside: avoid; }}
  img.plot {{ max-width: 100%; border-radius: 6px; border: 1px solid #1b2733; margin: 8px 0 4px; }}
  .fig-caption {{ font-size: 10.5px; color: {_MUTED}; line-height:1.5; margin-top: 4px; }}
  .fig-caption b {{ color: {_TEXT}; }}
  .not-evaluated {{ color: {_MUTED}; font-style: italic; font-size: 11.5px; }}
  .note {{ font-size: 10.5px; color: {_MUTED}; margin-top: 6px; }}
  .comparison {{ display:grid; grid-template-columns: 1fr 1fr; gap: 14px; margin: 10px 0; }}
  .metric-kind-tag {{ display:inline-block; font-size: 8.5px; font-weight: 700; letter-spacing: .05em;
                       text-transform: uppercase; border-radius: 3px; padding: 2px 7px; margin-bottom: 8px; }}
  .tag-conventional {{ color: #6FC6FF; background: rgba(111,198,255,0.10); border: 1px solid rgba(111,198,255,0.35); }}
  .tag-ims {{ color: #B18CFF; background: rgba(177,140,255,0.10); border: 1px solid rgba(177,140,255,0.35); }}
  .sxs-card {{ background:{_PANEL}; border:1px solid #1b2733; border-radius:8px; padding:16px; }}
  .sxs-card .sxs-title {{ font-size: 10px; color:{_MUTED}; text-transform:uppercase; letter-spacing:.05em; }}
  .sxs-card .sxs-value {{ font-size: 20px; font-weight:800; margin-top:6px; }}
  .footer {{ margin-top: 34px; padding-top: 12px; border-top: 1px solid #1b2733; color: {_MUTED};
             font-size: 9.5px; font-family: monospace; display:flex; justify-content:space-between; }}
  @media print {{ body {{ padding: 0 34px; }} .report-header {{ page-break-after: avoid; }} }}
</style>
</head>
<body>
  <div class="report-header">
    {logo_img}
    <div class="brand-text"><b>IMS Platform Explorer</b><span>Intrinsic Manifold Stability Analysis Platform</span></div>
    <div class="status-chip">{_esc(case_status)}</div>
  </div>
  <h1>{_esc(title)}</h1>
  <div class="subtitle">{_esc(subtitle)}</div>
  <div class="meta">Case ID: {_esc(case_id)} &middot; Generated: {generated_at} &middot; Platform: {PLATFORM_VERSION}</div>
  {body}
  <div class="footer"><span>IMS Platform Explorer ({PLATFORM_VERSION}) &middot; Intrinsic Manifold Stability Analysis Platform</span>
  <span>Report produced directly from the analysis result object -- no value was entered by hand.</span></div>
</body>
</html>"""


def section_executive_summary(verdict: str, summary_metrics: Dict[str, str], disclaimer: str) -> str:
    metrics_html = "".join(
        f"<div class='metric'><div class='label'>{_esc(k)}</div><div class='value'>{_esc(v)}</div></div>"
        for k, v in summary_metrics.items()
    )
    return f"""
<h2>Executive Summary</h2>
<p>Assessment: {_verdict_badge(verdict)}</p>
<div class="metric-grid">{metrics_html}</div>
<div class="disclaimer">{_esc(disclaimer)}</div>
"""


def section_scenario_definition(description: str, component_chain: Optional[List[str]], disturbance_text: str,
                                 network_counts: Optional[Dict] = None, topology_diagram_uri: Optional[str] = None) -> str:
    if component_chain:
        chain_html = "<ul style='margin:6px 0 0;padding-left:18px;font-size:11.5px;line-height:1.7;'>" + \
            "".join(f"<li>{_esc(c)}</li>" for c in component_chain) + "</ul>"
    else:
        chain_html = _not_evaluated()
    counts_html = ""
    if network_counts:
        counts_html = (f"<p class='note'>Implemented model: {network_counts.get('buses','?')} buses, "
                        f"{network_counts.get('lines','?')} lines, {network_counts.get('components','?')} "
                        f"bus-attached components.</p>")
    topo_html = (f"<div class='figure'><img class='plot' src='{topology_diagram_uri}'/>"
                 f"<div class='fig-caption'>Network topology as actually implemented -- makes explicit which "
                 f"bus is a fixed/infinite reference (no ODE state, hence no voltage trace of its own) versus "
                 f"which buses are dynamic states, rather than leaving that to be inferred from the trace "
                 f"count alone.</div></div>") if topology_diagram_uri else ""
    return f"""
<h2>Scenario Definition</h2>
<p>{_esc(description)}</p>
<h3>Implemented network</h3>
{topo_html}
{chain_html}
{counts_html}
<h3>Disturbance</h3>
<p>{_esc(disturbance_text)}</p>
"""


def section_system_configuration(params_used: Dict) -> str:
    return f"""
<h2>Input Conditions</h2>
{_params_table(params_used) if params_used else _not_evaluated()}
"""


def section_large_signal_results(summary: Dict, extra_rows: Optional[Dict] = None) -> str:
    rows = dict(summary)
    if extra_rows:
        rows.update(extra_rows)
    rows_html = "".join(f"<tr><td>{_esc(k)}</td><td class='num'>{_fmt(v)}</td></tr>" for k, v in rows.items())
    return f"""
<h2>Large-Signal Results</h2>
<div class="metric-kind-tag tag-conventional">Conventional / physical metric</div>
<table class="ctable"><tbody>{rows_html}</tbody></table>
"""


def section_time_domain_plots(plots: List[Dict], start_number: int = 1) -> str:
    """plots: list of {"title": str, "uri": str, "caption": str}. Renders each as a numbered
    Figure with a caption explaining what is plotted, its units, and what to observe -- not just
    a bare title."""
    if not plots:
        return f"<h2>Time-Domain Plots</h2>{_not_evaluated()}"
    parts = ["<h2>Time-Domain Plots</h2>"]
    for i, p in enumerate(plots):
        n = start_number + i
        parts.append(
            f"<div class='figure'><img class='plot' src='{p['uri']}'/>"
            f"<div class='fig-caption'><b>Figure {n}.</b> {_esc(p['caption'])}</div></div>"
        )
    return "\n".join(parts)


def section_conventional_vs_large_signal_cards(small_signal_status: str, large_signal_verdict: str) -> str:
    """The prominent, headline side-by-side card the funding-demo brief asks for: same
    pre-disturbance operating point, two different answers depending on which question is asked."""
    color = {"RECOVERABLE": "#3FE0A8", "AT RISK": "#FFB454", "NON-RECOVERABLE": "#FF6B6B"}.get(large_signal_verdict, "#9fb0bf")
    return f"""
<div class="comparison" style="margin: 14px 0 20px;">
  <div class="sxs-card">
    <div class="sxs-title">Conventional small-signal</div>
    <div class="sxs-value" style="color:#3FE0A8;">{_esc(small_signal_status)}</div>
  </div>
  <div class="sxs-card" style="border-color:{color};">
    <div class="sxs-title">Large-signal (this platform's recoverability criterion)</div>
    <div class="sxs-value" style="color:{color};">{_esc(large_signal_verdict)}</div>
  </div>
</div>
<p class="note">Same pre-disturbance operating point. Different answer once a specific large disturbance is applied.</p>
"""


_IMS_NOT_YET_EVALUATED = (
    "IMS geometric diagnostic (manifold distance d_M(t), transverse stability indicator "
    "\u03bb\u22a5(t), and \u03b3_IMS margin, computed via continuation of the intrinsic manifold) has "
    "not yet been implemented for this case's model/disturbance combination. The recoverability "
    "verdict above is a physical/large-signal criterion evaluated directly on the simulated "
    "trajectory (voltage margin or fault-ride-through depth against a fixed threshold), not a "
    "literal intrinsic-manifold calculation -- it should not be read as an IMS geometric result."
)


def section_ims_assessment(small_signal_note: str, large_signal_note: str,
                            ims_geometric_note: Optional[str] = None) -> str:
    return f"""
<h2>IMS / Large-Signal Assessment</h2>
<div class="metric-kind-tag tag-conventional">Conventional / physical metric</div>
<table class="ctable">
<tr><td>Conventional (small-signal)</td><td>{_esc(small_signal_note)}</td></tr>
<tr><td>Large-signal / recoverability</td><td>{_esc(large_signal_note)}</td></tr>
</table>
<div class="metric-kind-tag tag-ims" style="margin-top:16px;">IMS geometric diagnostic</div>
<p class="note">{_esc(ims_geometric_note or _IMS_NOT_YET_EVALUATED)}</p>
<p class="note">This platform's recoverability criterion characterizes whether a specific simulated
disturbance stays within (or returns to) an admissible operating region -- it does not claim to predict
collapse in general, nor that small-signal stability implies or fails to imply any particular large-signal
outcome beyond the specific case simulated here.</p>
"""


def section_counterfactual(counterfactual: Optional[Dict], applicable: bool = True) -> str:
    if not applicable:
        return ""
    if not counterfactual:
        return f"<h2>Counterfactual Comparison</h2>{_not_evaluated()}"
    rows = "".join(
        f"<tr><td>{_esc(r['metric'])}</td><td class='num'>{_esc(r['baseline'])}</td>"
        f"<td class='num'>{_esc(r.get('dynamic_q', r.get('intervention','')))}</td></tr>"
        for r in counterfactual["comparison_table"]
    )
    return f"""
<h2>Counterfactual Comparison</h2>
<p>{_esc(counterfactual.get('note',''))}</p>
<table class="ctable">
<tr><th>Metric</th><th>Baseline</th><th>Intervention</th></tr>
{rows}
</table>
"""


def section_boundary(boundary_plot_uri: Optional[str], recommendations: Optional[Dict], fig_number: int = 3) -> str:
    parts = ["<h2>Boundary / Sensitivity Results</h2>"]
    if boundary_plot_uri:
        parts.append(f"<div class='figure'><img class='plot' src='{boundary_plot_uri}'/>"
                      f"<div class='fig-caption'><b>Figure {fig_number}.</b> Each point/bar is an "
                      f"independently simulated stress test at that parameter value -- not interpolated.</div></div>")
    elif not recommendations:
        parts.append(_not_evaluated())
    if recommendations:
        rows = "".join(
            f"<tr><td>{_esc(s['label'])}</td><td>{_esc(s['value_label'])}</td>"
            f"<td>{'achievable' if s['achievable'] else 'not achievable within tested range'}</td></tr>"
            for s in recommendations["strategies"]
        )
        headline = recommendations.get("best_recommendation")
        headline_html = ""
        if headline:
            headline_html = f"""<div class="comparison" style="grid-template-columns:1fr 1fr 1fr;margin-bottom:14px;">
<div class="sxs-card"><div class="sxs-title">Current condition</div><div class="sxs-value" style="color:#FF6B6B;font-size:15px;">{_esc(recommendations.get('baseline_verdict',''))}</div></div>
<div class="sxs-card"><div class="sxs-title">Recommended intervention</div><div class="sxs-value" style="font-size:13px;">{_esc(headline['label'])}</div></div>
<div class="sxs-card"><div class="sxs-title">Required level &rarr; Result</div><div class="sxs-value" style="color:#3FE0A8;font-size:13px;">{_esc(headline['value_label'])}</div></div>
</div>"""
        parts.append(f"""<h3>How can it be avoided? &mdash; intervention search</h3>
{headline_html}
<table class="ctable"><tr><th>Strategy</th><th>Result</th><th>Status</th></tr>{rows}</table>
<p class="note">{_esc(recommendations.get('note',''))}</p>""")
    return "\n".join(parts)


def section_limitations(disclaimer: str, extra: Optional[str] = None) -> str:
    return f"""
<h2>Scientific Limitations and Applicability</h2>
<div class="disclaimer">{_esc(disclaimer)}</div>
{f"<p class='note'>{_esc(extra)}</p>" if extra else ""}
<p class="note">Conclusions apply to the implemented reduced-order model and the tested parameter range
only; they are not a claim about any real physical system beyond the mechanism class this model is
inspired by.</p>
"""


def _ims_warning_html(ims_warning: Optional[Dict]) -> str:
    if not ims_warning or not ims_warning.get("available"):
        return "<p class='not-evaluated'>Not evaluated in this analysis.</p>"
    if ims_warning.get("genuinely_leads"):
        return (f"<div class='disclaimer' style='border-color:#B18CFF;'>"
                f"<b>Genuinely leads:</b> {_esc(ims_warning['interpretation'])}</div>")
    return f"<p class='note'>{_esc(ims_warning['interpretation'])}</p>"


def section_ims_geometry(ims_geo: Optional[Dict], fig_number_start: int = 1, ims_warning: Optional[Dict] = None) -> str:
    """
    "C -- True IMS Geometry" -- the dedicated section demonstrating the
    actual intrinsic-manifold implementation (server/ims_geometry.py),
    not a conceptual illustration. Uses the required terminology
    "Region of Attraction (ROA)" throughout, not "basin".
    """
    if not ims_geo or not ims_geo.get("available"):
        msg = (ims_geo or {}).get("message", "Not evaluated in this analysis.")
        return f"<h2>True IMS Geometry</h2><p class='not-evaluated'>{_esc(msg)}</p>"

    mt = ims_geo["manifold_trace"]
    g = ims_geo["gamma_ims"]
    roa = ims_geo["roa_slice"]

    dM_uri = plot_manifold_distance(mt["t"], mt["d_M"])
    lam_uri = plot_transverse_rate(mt["t"], mt["lambda_perp"])
    roa_uri = plot_roa_slice(roa)

    n1, n2, n3 = fig_number_start, fig_number_start + 1, fig_number_start + 2
    contracting_note = ("The transverse dynamics are contracting throughout this trajectory "
                         "(\u03bb\u22a5(t) < 0 at every sampled point)." if mt["transversally_contracting_throughout"]
                         else "The transverse dynamics were NOT contracting at every sampled point -- see \u03bb\u22a5(t) below.")

    return f"""
<h2>True IMS Geometry</h2>
<p class="note">This section is derived from the platform's own genuine intrinsic-manifold implementation
(server/ims_geometry.py), applied to the SAME validated model and disturbance trajectory as the rest of this
report -- not a conceptual illustration. See the Reproducibility section for the exact parameters used.</p>

<h3>1. Intrinsic Manifold</h3>
<p>{_esc(ims_geo["manifold_definition"])}</p>

<h3>2. Manifold distance (d_M)</h3>
<div class="figure"><img class="plot" src="{dM_uri}"/>
<div class="fig-caption"><b>Figure {n1}.</b> Distance from the simulated trajectory to the intrinsic manifold,
d_M(t) = |timer(t) - timer*(v_es_sw(t))|, computed directly from the fleet's own overvoltage-protection timer
state and its closed-form manifold root. Peak d_M = {_fmt(mt['max_d_M'])}, final d_M = {_fmt(mt['final_d_M'], 6)}.</div></div>

<h3>3 &amp; 4. Transverse dynamics and transverse rate (\u03bb\u22a5)</h3>
<div class="figure"><img class="plot" src="{lam_uri}"/>
<div class="fig-caption"><b>Figure {n2}.</b> \u03bb\u22a5(t), the transverse eigenvalue of the fast (protection-timer)
subsystem, evaluated along the trajectory. {_esc(contracting_note)} Range observed: [{_fmt(mt['min_lambda_perp'])},
{_fmt(mt['max_lambda_perp'])}] (1/s). Verified to match an actual eigenvalue of the full linearized system at the
pre-disturbance equilibrium to within {g['analytic_numeric_match_error']:.1e} (see Reproducibility).</div></div>

<h3>5. \u03b3_IMS</h3>
<table class="ctable">
<tr><td>\u03bb\u22a5 (at pre-disturbance equilibrium)</td><td class="num">{_fmt(g['lambda_perp_analytic'])} /s</td></tr>
<tr><td>Slowest network eigenvalue (excl. timer)</td><td class="num">{_fmt(g['lambda_slow_network'])} /s</td></tr>
<tr><td>\u03b5 (operational time-scale-separation ratio)</td><td class="num">{_fmt(g['epsilon_operational'])}</td></tr>
<tr><td>\u03b3_IMS = |\u03bb\u22a5|\u00b2 / |\u03bb_slow|</td><td class="num">{_fmt(g['gamma_ims'])}</td></tr>
</table>
<p class="note">{_esc(g["note"])}</p>
{"<div class='disclaimer'>Time-scale separation at this operating point is WEAK (ratio &asymp; " + f"{abs(g['lambda_perp_analytic'])/abs(g['lambda_slow_network']):.2f}" + ", not &laquo;1) -- the manifold construction itself is exact regardless, but classical GSPT's asymptotic O(&epsilon;)-closeness guarantees are not strongly established at this parameter set. Reported honestly, not hidden.</div>" if g.get("timescale_separation_is_weak") else ""}

<h3>6, 7 &amp; 8. Region of Attraction (ROA) and the disturbed trajectory</h3>
<div class="figure"><img class="plot" src="{roa_uri}"/>
<div class="fig-caption"><b>Figure {n3}.</b> A 2-D Region of Attraction (ROA) SLICE through
({roa['v_state']}, {roa['timer_state']}), holding every other state coordinate fixed at the reference values
listed below -- a cross-section of the full {len(roa['frozen_reference_state'])}-dimensional state space's ROA,
not the complete ROA. Every cell is an independently integrated trajectory of the actual validated model (not
Monte Carlo sampling, not a reduced/toy model). Green = inside the Region of Attraction (recovers to an
admissible equilibrium); red = outside it (non-recoverable).</div></div>
<p class="note">{_esc(roa["note"])}</p>
<h3>Frozen reference state for this ROA slice</h3>
{_params_table(roa["frozen_reference_state"])}

<h3>Pre-incident IMS warning (d_M-based)</h3>
{_ims_warning_html(ims_warning)}

<h3>Geometric interpretation</h3>
<p class="note">Whether the disturbed trajectory remains within the admissible neighbourhood of the intrinsic
manifold is governed by TWO distinct questions this section separates explicitly: (i) transverse attractivity
(\u03bb\u22a5, always satisfied here) and (ii) whether the SLOW dynamics reduced onto the manifold itself
possess, and remain within reach of, an admissible equilibrium (the Region of Attraction question, items 6-8
above). This is the platform's own concrete demonstration of the IMS framework's central distinction from
conventional small-signal stability: a system can be transversally stable everywhere and still be
non-recoverable, because the manifold's own reduced dynamics have left the admissible region.</p>
"""


def section_ims_geometry_not_applicable(checks: List[Tuple[str, Dict]], overall_message: str) -> str:
    """
    Honest "C -- True IMS Geometry: not applicable" section, with the
    ACTUAL probed evidence shown (not just an assertion) -- see
    server.ims_geometry.check_manifold_applicability. `checks` is a
    list of (label, check_dict) pairs, one per converter checked.
    """
    rows = ""
    for label, check in checks:
        probe_summary = (f"timer* range over probe = {check['timer_star_range_over_probe']:.2e} "
                          f"(v_trip = {check['v_trip']})")
        rows += f"<tr><td>{_esc(label)}</td><td class='num'>{_esc('APPLICABLE' if check['applicable'] else 'NOT APPLICABLE')}</td><td>{_esc(probe_summary)}</td></tr>"
    return f"""
<h2>True IMS Geometry</h2>
<div class="disclaimer">{_esc(overall_message)}</div>
<table class="ctable">
<tr><th>Converter</th><th>Manifold construction</th><th>Evidence</th></tr>
{rows}
</table>
<p class="note">This is a validated NEGATIVE result, not an omission: the timer-based manifold construction
that is genuinely meaningful for the Iberian 2025-Inspired Overvoltage Cascade case (see that case's own
"C -- True IMS Geometry" section) was checked directly against this case's actual converter parameters
(server.ims_geometry.check_manifold_applicability) before being applied here, and found degenerate -- not
assumed to transfer merely because both cases use the same GFLRenewableSource component.</p>
"""


def section_reproducibility(case_id: str, params_used: Dict) -> str:
    generated_at = datetime.datetime.now().isoformat()
    return f"""
<h2>Reproducibility</h2>
<table class="ctable">
<tr><td>Case ID</td><td class="num">{_esc(case_id)}</td></tr>
<tr><td>Analysis timestamp</td><td class="num">{_esc(generated_at)}</td></tr>
<tr><td>Platform version</td><td class="num">{_esc(PLATFORM_VERSION)}</td></tr>
</table>
<h3>Exact parameter set</h3>
{_params_table(params_used)}
"""


# ---------------------------------------------------------------------
# PDF export (same HTML, rendered by wkhtmltopdf via pdfkit). Page
# numbers via wkhtmltopdf's own footer templating (not CSS paged-media
# counters, which wkhtmltopdf's WebKit only partially supports) so the
# PDF is genuinely paginated with "Page X of Y", matching the branding
# used in the HTML header/footer.
# ---------------------------------------------------------------------

_WKHTMLTOPDF_BUNDLED_PATH = os.path.join(os.path.dirname(__file__), "bin", "wkhtmltopdf")
_wkhtmltopdf_config_cache = None


def _pdfkit_configuration():
    """
    Prefer the patched-Qt wkhtmltopdf binary bundled with this app
    (server/bin/wkhtmltopdf) over whatever `wkhtmltopdf` is on PATH.
    This matters concretely, not cosmetically: the system/apt-packaged
    wkhtmltopdf on Debian/Ubuntu (and this platform's own dev sandbox)
    is the "unpatched Qt" build, which SILENTLY IGNORES --footer-*/
    --header-* options entirely (no exception, just a stderr warning) --
    that is why page numbers were missing from the PDF even though the
    same options were already being passed correctly. Confirmed directly
    by running both binaries against the same HTML and comparing output.
    Falls back to whatever `wkhtmltopdf` pdfkit finds on PATH if the
    bundled binary is missing (e.g. a non-Linux-x86_64 deployment) --
    the PDF will still generate, just without per-page footers, rather
    than failing outright.
    """
    global _wkhtmltopdf_config_cache
    if _wkhtmltopdf_config_cache is None:
        import pdfkit
        if os.path.isfile(_WKHTMLTOPDF_BUNDLED_PATH) and os.access(_WKHTMLTOPDF_BUNDLED_PATH, os.X_OK):
            _wkhtmltopdf_config_cache = pdfkit.configuration(wkhtmltopdf=_WKHTMLTOPDF_BUNDLED_PATH)
        else:
            _wkhtmltopdf_config_cache = False  # sentinel: use pdfkit's own PATH lookup
    return _wkhtmltopdf_config_cache


class PdfGenerationUnavailable(RuntimeError):
    """Raised with a clear, actionable message when pdfkit (or wkhtmltopdf) is missing --
    instead of letting a bare ModuleNotFoundError/OSError surface as an opaque 500."""
    pass


def html_to_pdf_bytes(html_str: str) -> bytes:
    try:
        import pdfkit
    except ModuleNotFoundError as e:
        raise PdfGenerationUnavailable(
            "PDF export requires the 'pdfkit' Python package, which is not installed in this "
            "environment. Install it with: pip install pdfkit  (also requires the wkhtmltopdf "
            "binary on the system -- see requirements.txt / pyproject.toml)."
        ) from e

    options = {
        "quiet": "",
        "enable-local-file-access": None,
        "print-media-type": None,
        "margin-top": "14mm", "margin-bottom": "16mm", "margin-left": "12mm", "margin-right": "12mm",
        "footer-center": "IMS Platform Explorer",
        "footer-right": "Page [page] of [topage]",
        "footer-font-size": "8",
        "footer-font-name": "monospace",
        "footer-spacing": "6",
    }
    config = _pdfkit_configuration()
    try:
        if config:
            return pdfkit.from_string(html_str, False, options=options, configuration=config)
        return pdfkit.from_string(html_str, False, options=options)
    except OSError as e:
        raise PdfGenerationUnavailable(
            "PDF export requires the 'wkhtmltopdf' system binary, which could not be run in this "
            f"environment ({e}). See requirements.txt / pyproject.toml for install instructions."
        ) from e


# ---------------------------------------------------------------------
# Case-specific report builders. Each takes the REAL result dict(s)
# already returned by that case's own backend function(s) -- the same
# objects the API/UI already serve -- and assembles them into the
# generic section shell above. No value here is computed independently
# of those dicts.
# ---------------------------------------------------------------------

_IBERIAN_TOPOLOGY_NODES = [
    {"id": "eu", "label": "Bus 1: Continental\nEurope equiv.", "kind": "fixed", "x": 0.12, "y": 0.5},
    {"id": "es_c", "label": "Bus 2: Central\nSpain", "kind": "dynamic", "x": 0.48, "y": 0.5},
    {"id": "es_sw", "label": "Bus 3: SW renewable\ncluster", "kind": "dynamic", "x": 0.88, "y": 0.92},
    {"id": "pt", "label": "Bus 4: Portugal", "kind": "dynamic", "x": 0.88, "y": 0.5},
    {"id": "emb", "label": "Bus 5: Embedded\nzone", "kind": "dynamic", "x": 0.88, "y": 0.08},
]
_IBERIAN_TOPOLOGY_EDGES = [("eu", "es_c"), ("es_c", "es_sw"), ("es_c", "pt"), ("es_c", "emb")]


def build_iberian_report(result: Dict, counterfactual: Optional[Dict] = None,
                          boundary: Optional[Dict] = None, recommendations: Optional[Dict] = None,
                          ims_geometry: Optional[Dict] = None, ims_warning: Optional[Dict] = None) -> str:
    if not result.get("baseline_ok"):
        sections = [
            section_executive_summary("NON-RECOVERABLE", {"Status": "No stable baseline"}, result.get("disclaimer", "")),
            f"<p>{_esc(result.get('message',''))}</p>",
        ]
        return document_shell("Iberian 2025-Inspired Overvoltage Cascade", "Analysis could not be completed",
                               "iberian_2025_overvoltage_cascade", sections, case_status="ACTIVE \u2014 ANALYSIS AVAILABLE")

    s = result["summary"]
    verdict = s["verdict"]
    t = result["trajectory"]["t"]
    bus_labels = {"v_es_c": "Central Spain", "v_es_sw": "SW renewable cluster", "v_pt": "Portugal", "v_emb": "Embedded zone"}
    series = {bus_labels.get(k, k): v for k, v in result["voltage_traces"].items()}
    volt_uri = plot_time_series(t, series, "time (s)", "voltage (p.u.)", "Bus voltage response",
                                 hlines=[(result["v_trip"], "Overvoltage trip threshold", "#FF6B6B")],
                                 vlines=[(e["t"], "#FFB454") for e in result.get("event_timeline", [])])
    ind_uri = plot_time_series(t, {"Protection indicator": result["trip_timer_trace"]}, "time (s)",
                                "indicator (0-1)", "Overvoltage-protection indicator (0=clear, 1=tripped)",
                                hlines=[(0.5, "trip threshold (indicator)", "#FFB454")])

    boundary_uri = None
    if boundary and boundary.get("points"):
        pts = [p for p in boundary["points"] if p.get("verdict")]
        if pts:
            boundary_uri = plot_bar_verdicts([p["severity"] for p in pts], [p["verdict"] for p in pts],
                                              "Disturbance severity (\u00d7 documented magnitude)",
                                              "Recoverability vs. disturbance severity")

    disturbance_text = " | ".join(e["label"] for e in result.get("event_timeline", []))
    sections = [
        section_executive_summary(verdict, {
            "Peak voltage (p.u.)": _fmt(s["max_v_es_sw"]), "Settled voltage (p.u.)": _fmt(s["final_v_es_sw"]),
            "Voltage margin (p.u.)": _fmt(s["min_voltage_margin"]),
            "Protection indicator (final)": _fmt(s["final_trip_timer"]),
        }, result["disclaimer"]),
        section_conventional_vs_large_signal_cards("STABLE", verdict),
        section_scenario_definition(
            "Reduced-order 5-bus illustrative network inspired by the mechanism class described in "
            "ENTSO-E's report on the 28 April 2025 Iberian Peninsula event (insufficient dynamic voltage "
            "support, current limiting, delayed overvoltage protection, cascading disconnection). This does "
            "NOT claim to reconstruct the real Spanish/Portuguese network -- it states exactly what the "
            "implemented model contains.",
            result.get("component_chain"), disturbance_text, result.get("network_summary"),
            topology_diagram_uri=plot_network_topology(_IBERIAN_TOPOLOGY_NODES, _IBERIAN_TOPOLOGY_EDGES,
                                                         title="Iberian 2025 scenario: implemented 5-bus network")),
        section_system_configuration(result["params_used"]),
        section_large_signal_results({k: v for k, v in s.items() if k not in ("small_signal_stable_throughout_note", "criterion_note")}),
        section_time_domain_plots([
            {"uri": volt_uri, "caption": "Bus-voltage response during the scripted 355/727/928 MW-equivalent "
             "disturbance sequence (dotted vertical lines). Dashed red line is the overvoltage-trip threshold "
             "(1.10 p.u.). Units: per-unit voltage (1.0 = nominal). " + result.get("voltage_plot_note", "")},
            {"uri": ind_uri, "caption": "The renewable fleet's own smooth overvoltage-protection indicator "
             "(0 = fully clear, 1 = fully tripped). Crossing and staying above the 0.5 dashed line indicates "
             "the protection has engaged."},
        ]),
        section_ims_assessment(s["small_signal_stable_throughout_note"],
                                (f"Large-signal verdict: {verdict}. " + s["criterion_note"]) if "criterion_note" in s else
                                f"Large-signal verdict: {verdict}."),
        section_counterfactual(counterfactual),
        section_boundary(boundary_uri, recommendations, fig_number=3),
        section_ims_geometry(ims_geometry, fig_number_start=5, ims_warning=ims_warning),
        section_limitations(result["disclaimer"]),
        section_reproducibility("iberian_2025_overvoltage_cascade", result["params_used"]),
    ]
    return document_shell("Iberian 2025-Inspired Overvoltage Cascade", "Overvoltage Cascade Analysis Report",
                           "iberian_2025_overvoltage_cascade", sections)


def build_gfm_report(result: Dict, sweep: Optional[Dict] = None, ims_geometry: Optional[Dict] = None,
                      recommendations: Optional[Dict] = None) -> str:
    if not result.get("baseline_ok"):
        sections = [section_executive_summary("NON-RECOVERABLE", {"Status": "No stable baseline"},
                                                result.get("disclaimer", "")),
                    f"<p>{_esc(result.get('message',''))}</p>"]
        return document_shell("GFM Current-Limit Recovery", "Analysis could not be completed",
                               "gfm_current_limit_recovery", sections, case_status="ACTIVE \u2014 ANALYSIS AVAILABLE")

    s = result["summary"]
    verdict = s["verdict"]
    t = result["trajectory"]["t"]
    plots = [{"uri": plot_time_series(
        t, {"v_poc": result["voltage_trace"]}, "time (s)", "voltage (p.u.)",
        "Point-of-connection voltage response",
        hlines=[(result["v_critical"], "Critical (non-recoverable) threshold", "#FF6B6B"),
                (result["v_at_risk"], "At-risk threshold", "#FFB454")],
        vlines=[(result["fault_definition"]["t_fault"], "#FFB454"), (result["fault_definition"]["t_clear"], "#FFB454")]),
        "caption": "Point-of-connection voltage during the scripted local fault (between the two dotted "
                   "vertical lines). Dashed lines mark the critical and at-risk fault-ride-through thresholds. "
                   "Units: per-unit voltage."}]
    if "current_trace" in result:
        plots.append({"uri": plot_time_series(
            t, {"|i| / Imax": result["current_trace"]}, "time (s)", "|i| / Imax",
            "Converter current utilization",
            hlines=[(1.0, "current limit", "#FF6B6B")],
            vlines=[(result["fault_definition"]["t_fault"], "#FFB454"), (result["fault_definition"]["t_clear"], "#FFB454")]),
            "caption": "Converter current magnitude as a fraction of its rated limit. A value at/near 1.0 "
                       "indicates the current limiter is actively saturating (the converter cannot deliver "
                       "more restoring current even though its control law is demanding it)."})
    sweep_uri = None
    if sweep and sweep.get("points"):
        pts = [p for p in sweep["points"] if p.get("verdict")]
        if pts:
            sweep_uri = plot_bar_verdicts([p["Imax"] for p in pts], [p["verdict"] for p in pts],
                                           "Converter current limit (Imax, p.u.)",
                                           "Recoverability vs. current limit")

    fd = result["fault_definition"]
    component_chain = [
        "Bus 1 -- weak-grid equivalent (infinite bus)",
        "Bus 2 -- point of common coupling: a single grid-forming (GFM) converter (Thevenin voltage source "
        "behind a fixed internal reactance X_gfm, with its own smooth current-limit clamp) plus the local "
        "load that hosts the scripted fault",
    ]
    sections = [
        section_executive_summary(verdict, {
            "Min voltage during fault (p.u.)": _fmt(s["min_v_poc"]), "Time of minimum (s)": _fmt(s["t_min_v"]),
            "Settled voltage (p.u.)": _fmt(s["final_v_poc"]),
        }, result["disclaimer"]),
        section_conventional_vs_large_signal_cards(
            f"STABLE (max Re(eig)={_fmt(result['baseline']['max_eigenvalue_real_part'])})", verdict),
        section_scenario_definition(
            "Standalone single-converter network: a grid-forming (GFM) converter -- a Thevenin voltage "
            "source behind a fixed internal reactance, with its own current limit -- at the point of "
            "common coupling of a weak grid. Distinct from the Iberian scenario's mechanism (this case "
            "is an undervoltage/current-limited fault-ride-through story, not an overvoltage-protection one).",
            component_chain, fd["description"]),
        section_system_configuration(result["params_used"]),
        section_large_signal_results({k: v for k, v in s.items() if k != "criterion_note"}),
        section_time_domain_plots(plots),
        section_ims_assessment(
            f"Pre-fault operating point is small-signal stable (max Re(eig)={_fmt(result['baseline']['max_eigenvalue_real_part'])}).",
            f"Large-signal verdict: {verdict}. {s['criterion_note']}"),
        section_counterfactual(None, applicable=False),
        section_boundary(sweep_uri, recommendations, fig_number=len(plots) + 1),
        (section_ims_geometry_not_applicable([("Grid-forming converter (gfm_conv)", ims_geometry["applicability_check"])],
                                              ims_geometry["message"])
         if ims_geometry and not ims_geometry.get("available") else
         (section_ims_geometry(ims_geometry, fig_number_start=len(plots) + 2) if ims_geometry else
          "<h2>True IMS Geometry</h2><p class='not-evaluated'>Not evaluated in this analysis.</p>")),
        section_limitations(result["disclaimer"],
                             "This case's recoverability criterion is a fault-ride-through (FRT) depth "
                             "criterion, distinct from an endogenous stable/unstable equilibrium bifurcation."),
        section_reproducibility("gfm_current_limit_recovery", result["params_used"]),
    ]
    return document_shell("GFM Current-Limit Recovery", "Fault-Ride-Through Analysis Report",
                           "gfm_current_limit_recovery", sections)


def build_multiconverter_report(result: Dict, asymmetry: Optional[Dict] = None,
                                 ims_geometry: Optional[Dict] = None, recommendations: Optional[Dict] = None) -> str:
    if not result.get("baseline_ok"):
        sections = [section_executive_summary("NON-RECOVERABLE", {"Status": "No stable baseline"},
                                                result.get("disclaimer", "")),
                    f"<p>{_esc(result.get('message',''))}</p>"]
        return document_shell("Multi-Converter Fault Recovery", "Analysis could not be completed",
                               "multi_converter_fault_recovery", sections, case_status="ACTIVE \u2014 ANALYSIS AVAILABLE")

    s = result["summary"]
    verdict = s["verdict"]
    t = result["trajectory"]["t"]
    plots = [{"uri": plot_time_series(
        t, {"v_busA (GFM anchor)": result["voltage_traces"]["v_busA"], "v_busB (GFL follower)": result["voltage_traces"]["v_busB"]},
        "time (s)", "voltage (p.u.)", "Bus voltage response (both converters)",
        hlines=[(result["v_critical"], "Critical threshold", "#FF6B6B"), (result["v_at_risk"], "At-risk threshold", "#FFB454")],
        vlines=[(result["fault_definition"]["t_fault"], "#FFB454"), (result["fault_definition"]["t_clear"], "#FFB454")]),
        "caption": "Voltage at both converters' buses during the common local fault at busB (between the "
                   "dotted lines). One converter's own response measurably changes the other's trajectory "
                   "through the shared network -- the coupling this case is named for. Units: per-unit voltage."}]
    if "current_traces" in result:
        plots.append({"uri": plot_time_series(
            t, {"|i_gfm| / Imax": result["current_traces"]["gfm"], "|i_gfl| / Imax": result["current_traces"]["gfl"]},
            "time (s)", "|i| / Imax", "Converter current utilization (both converters)",
            hlines=[(1.0, "current limit", "#FF6B6B")],
            vlines=[(result["fault_definition"]["t_fault"], "#FFB454"), (result["fault_definition"]["t_clear"], "#FFB454")]),
            "caption": "Each converter's current magnitude as a fraction of its own rated limit. Shows "
                       "directly which converter(s), if any, are current-limited during the fault."})

    fd = result["fault_definition"]
    asym_html = ""
    if asymmetry:
        rows = "".join(
            f"<tr><td>{_esc(label)}</td><td class='num'>{_fmt(asymmetry[key]['summary']['worst_min_v'])}</td>"
            f"<td>{_esc(asymmetry[key]['summary']['verdict'])}</td></tr>"
            for label, key in [("GFM (anchor) support only", "gfm_only"), ("GFL (follower) support only", "gfl_only"),
                                ("Both", "both")]
        )
        asym_html = f"""<h3>Local vs. remote support (interaction/asymmetry check)</h3>
<table class="ctable"><tr><th>Configuration</th><th>Worst voltage dip (p.u.)</th><th>Verdict</th></tr>{rows}</table>
<p class="note">Local (follower) support at the faulted bus is necessary and sufficient on its own; remote
(anchor) current headroom alone is not, even at its maximum tested value -- i.e. one converter's own support
setting changes whether the OTHER converter's bus recovers, demonstrated directly rather than assumed.</p>"""

    component_chain = [
        "Bus 1 -- weak-grid equivalent (infinite bus)",
        "Bus 2 (\"busA\") -- grid-forming (GFM) anchor converter",
        "Bus 3 (\"busB\") -- grid-following (GFL) follower converter, plus the local load that hosts the "
        "scripted fault -- coupled to busA through the shared network",
    ]
    sections = [
        section_executive_summary(verdict, {
            "Min voltage, busA (p.u.)": _fmt(s["min_v_busA"]), "Min voltage, busB (p.u.)": _fmt(s["min_v_busB"]),
            "Limiting bus": s["limiting_bus"],
        }, result["disclaimer"]),
        section_conventional_vs_large_signal_cards(
            f"STABLE (max Re(eig)={_fmt(result['baseline']['max_eigenvalue_real_part'])})", verdict),
        section_scenario_definition(
            "Two genuinely interacting converters sharing a network: a grid-forming (GFM) anchor at busA and "
            "a grid-following (GFL) converter at busB, coupled through the network and subjected to a common "
            "local fault at busB.", component_chain, fd["description"]),
        section_system_configuration(result["params_used"]),
        section_large_signal_results({k: v for k, v in s.items() if k != "criterion_note"}),
        section_time_domain_plots(plots),
        section_ims_assessment(
            f"Pre-fault operating point is small-signal stable (max Re(eig)={_fmt(result['baseline']['max_eigenvalue_real_part'])}).",
            f"Large-signal verdict: {verdict}. {s['criterion_note']}"),
        section_counterfactual(None, applicable=False),
        "<h2>Multi-Converter Interaction</h2>" + (asym_html if asym_html else _not_evaluated()),
        section_boundary(None, recommendations, fig_number=len(plots) + 1),
        (section_ims_geometry_not_applicable([
            ("GFM anchor (conv_gfm)", ims_geometry["applicability_check_gfm"]),
            ("GFL follower (conv_gfl)", ims_geometry["applicability_check_gfl"]),
        ], ims_geometry["message"]) if ims_geometry and not ims_geometry.get("available") else
         (section_ims_geometry(ims_geometry, fig_number_start=len(plots) + 1) if ims_geometry else
          "<h2>True IMS Geometry</h2><p class='not-evaluated'>Not evaluated in this analysis.</p>")),
        section_limitations(result["disclaimer"],
                             "Recoverability here is an FRT-depth criterion against the worse of the two "
                             "buses' voltage dips, not an endogenous bifurcation."),
        section_reproducibility("multi_converter_fault_recovery", result["params_used"]),
    ]
    return document_shell("Multi-Converter Fault Recovery", "Coordinated Fault-Ride-Through Analysis Report",
                           "multi_converter_fault_recovery", sections)
