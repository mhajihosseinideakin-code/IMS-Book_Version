"""
reporting.report
-----------------

Automated engineering-report generation, turning an IMS/MRC analysis
run into a shareable Markdown or self-contained HTML summary (the
"Reports" capability of the IMS Platform dashboard).
"""

from __future__ import annotations

import datetime as _dt
import os
from typing import Optional, List

from ..core.system import DynamicalSystem
from ..ims.manifold import IntrinsicManifold
from ..ims.recoverability import RecoverabilityReport


def generate_markdown_report(
    system: DynamicalSystem,
    manifold: IntrinsicManifold,
    recoverability: RecoverabilityReport,
    case_name: str = "IMS Case Study",
    controller_name: Optional[str] = None,
    extra_notes: Optional[str] = None,
) -> str:
    """Produce a Markdown engineering report string summarising an IMS/MRC run."""
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    n_stable = int(sum(p.is_stable for p in manifold.points))
    n_total = len(manifold.points)

    lines = []
    lines.append(f"# IMS Analysis Report — {case_name}")
    lines.append(f"*Generated {ts} by IMS Platform (core engine)*\n")

    lines.append("## System Model")
    lines.append(f"- Model: `{system.__class__.__name__}` (\"{system.name}\")")
    lines.append(f"- States: {', '.join(system.state_names)}")
    lines.append(f"- Inputs: {', '.join(system.input_names)}")
    lines.append(f"- Parameters: `{system.params}`\n")

    lines.append("## Intrinsic Manifold")
    lines.append(f"- Sweep parameter: `{manifold.param_name}`")
    lines.append(f"- Manifold samples: {n_total} ({n_stable} stable / {n_total - n_stable} unstable)\n")

    lines.append("## Recoverability Assessment")
    lines.append(f"- Nominal state: `{recoverability.nominal_state}`")
    lines.append(f"- Scenarios analysed: {recoverability.n_samples}")
    lines.append(f"- Recoverable scenarios: {recoverability.n_recoverable} "
                 f"({100*recoverability.recoverability_index:.1f}%)")
    lines.append(f"- **Recoverability Index: {recoverability.recoverability_index:.2f}**")
    lines.append(f"- Margin to critical boundary: {recoverability.margin_to_boundary:.4f}")
    lines.append(f"- **Risk level: {recoverability.risk_level}**\n")

    if controller_name:
        lines.append("## Control Strategy")
        lines.append(f"- Controller: {controller_name}")
        lines.append("- Design objective: Manifold-Reshaping Control (MRC), driving trajectories "
                      "back onto the intrinsic manifold following large disturbances.\n")

    lines.append("## Engineering Interpretation")
    if recoverability.risk_level == "LOW":
        interp = ("The system exhibits a large recoverability region relative to the sampled "
                  "disturbance envelope; nominal operation is well clear of the critical boundary.")
    elif recoverability.risk_level == "MEDIUM":
        interp = ("A non-trivial fraction of sampled disturbances are non-recoverable. Consider "
                  "tightening protection settings, increasing droop/damping, or applying "
                  "Manifold-Reshaping Control to enlarge the recoverable region.")
    else:
        interp = ("The majority of sampled disturbances are non-recoverable at this operating "
                  "point. This operating condition should be treated as high risk; recommend "
                  "re-dispatch, controller re-tuning, or MRC-based trajectory reshaping before "
                  "relying on this operating point under contingency conditions.")
    lines.append(interp + "\n")

    if extra_notes:
        lines.append("## Notes")
        lines.append(extra_notes + "\n")

    lines.append("---")
    lines.append("*This report was generated automatically by the IMS Platform core engine. "
                 "IMS complements, and does not replace, established equilibrium-based stability "
                 "analysis (small-signal, transient stability, eigenvalue/Lyapunov methods).*")

    return "\n".join(lines)


def generate_html_report(
    system: DynamicalSystem,
    manifold: IntrinsicManifold,
    recoverability: RecoverabilityReport,
    case_name: str = "IMS Case Study",
    controller_name: Optional[str] = None,
    extra_notes: Optional[str] = None,
    image_dir: Optional[str] = None,
    image_names: Optional[List[str]] = None,
) -> str:
    """
    Produce a self-contained HTML report (referencing plot images by
    relative filename, so it should be written into the same directory
    as the images) summarising an IMS/MRC run.
    """
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    n_stable = int(sum(p.is_stable for p in manifold.points))
    n_total = len(manifold.points)
    image_names = image_names or []

    risk_colors = {"LOW": "#2e7d32", "MEDIUM": "#f9a825", "HIGH": "#c62828"}
    risk_color = risk_colors.get(recoverability.risk_level, "#616161")

    if recoverability.risk_level == "LOW":
        interp = ("The system exhibits a large recoverability region relative to the sampled "
                  "disturbance envelope; nominal operation is well clear of the critical boundary.")
    elif recoverability.risk_level == "MEDIUM":
        interp = ("A non-trivial fraction of sampled disturbances are non-recoverable. Consider "
                  "tightening protection settings, increasing droop/damping, or applying "
                  "Manifold-Reshaping Control to enlarge the recoverable region.")
    else:
        interp = ("The majority of sampled disturbances are non-recoverable at this operating "
                  "point. Recommend re-dispatch, controller re-tuning, or MRC-based trajectory "
                  "reshaping before relying on this operating point under contingency conditions.")

    images_html = "\n".join(
        f'<div class="fig"><img src="{name}" alt="{name}"></div>' for name in image_names
    )

    controller_html = ""
    if controller_name:
        controller_html = f"""
        <h2>Control Strategy</h2>
        <p><b>Controller:</b> {controller_name}</p>
        <p>Design objective: Manifold-Reshaping Control (MRC), driving trajectories back onto the
        intrinsic manifold following large disturbances.</p>
        """

    notes_html = f"<h2>Notes</h2><p>{extra_notes}</p>" if extra_notes else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>IMS Report — {case_name}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
          max-width: 880px; margin: 40px auto; padding: 0 20px; color: #212121; line-height: 1.55; }}
  h1 {{ font-size: 1.6em; border-bottom: 3px solid #2e7d32; padding-bottom: 8px; }}
  h2 {{ font-size: 1.2em; margin-top: 2em; color: #1b5e20; }}
  .meta {{ color: #757575; font-size: 0.9em; }}
  .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                   gap: 12px; margin: 16px 0; }}
  .metric {{ background: #f5f5f5; border-radius: 8px; padding: 14px; }}
  .metric .label {{ font-size: 0.8em; color: #616161; text-transform: uppercase; letter-spacing: 0.03em; }}
  .metric .value {{ font-size: 1.5em; font-weight: 600; margin-top: 4px; }}
  .risk-badge {{ display: inline-block; padding: 3px 12px; border-radius: 999px; color: white;
                  font-weight: 600; font-size: 0.85em; background: {risk_color}; }}
  .fig {{ margin: 18px 0; text-align: center; }}
  .fig img {{ max-width: 100%; border: 1px solid #e0e0e0; border-radius: 6px; }}
  code {{ background: #f0f0f0; padding: 1px 6px; border-radius: 4px; font-size: 0.9em; }}
  footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #e0e0e0; color: #9e9e9e; font-size: 0.85em; }}
</style>
</head>
<body>
  <h1>IMS Analysis Report — {case_name}</h1>
  <p class="meta">Generated {ts} by IMS Platform (core engine)</p>

  <h2>System Model</h2>
  <p><b>Model:</b> <code>{system.__class__.__name__}</code> ("{system.name}")<br>
     <b>States:</b> {', '.join(system.state_names)}<br>
     <b>Inputs:</b> {', '.join(system.input_names)}</p>

  <h2>Intrinsic Manifold</h2>
  <p><b>Sweep parameter:</b> <code>{manifold.param_name}</code><br>
     <b>Manifold samples:</b> {n_total} ({n_stable} stable / {n_total - n_stable} unstable)</p>

  <h2>Recoverability Assessment</h2>
  <div class="metric-grid">
    <div class="metric"><div class="label">Recoverability Index</div>
      <div class="value">{recoverability.recoverability_index:.2f}</div></div>
    <div class="metric"><div class="label">Recoverable Scenarios</div>
      <div class="value">{recoverability.n_recoverable}/{recoverability.n_samples}</div></div>
    <div class="metric"><div class="label">Margin to Boundary</div>
      <div class="value">{recoverability.margin_to_boundary:.4f}</div></div>
    <div class="metric"><div class="label">Risk Level</div>
      <div class="value"><span class="risk-badge">{recoverability.risk_level}</span></div></div>
  </div>
  {controller_html}
  <h2>Engineering Interpretation</h2>
  <p>{interp}</p>
  {notes_html}
  <h2>Figures</h2>
  {images_html}
  <footer>
    This report was generated automatically by the IMS Platform core engine. IMS complements,
    and does not replace, established equilibrium-based stability analysis (small-signal,
    transient stability, eigenvalue/Lyapunov methods).
  </footer>
</body>
</html>
"""
    return html
