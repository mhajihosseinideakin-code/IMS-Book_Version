"""
ims_platform.stabilizing.report
--------------------------------

PDF report for the ideal four-state stabilizing-MRC run, generated with
matplotlib (figures) + reportlab (document). Built from the SAME verified
dataset used by the UI plots and CSV export -- no trajectory is
re-simulated for reporting.

The report explicitly states the guarantee classification: exact residual
contraction belongs to the IDEAL UNSATURATED model, and local linear
stability does NOT by itself constitute a certified regional IMS
guarantee.
"""

from __future__ import annotations

import io
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
)

from .workflow import compute_equilibrium, compute_stability, PARAM_SCHEMA, MODEL_ID, MODEL_VERSION


def _fmt(z: Dict) -> str:
    if abs(z["im"]) < 1e-9:
        return f"{z['re']:.4g}"
    sign = "+" if z["im"] >= 0 else "-"
    return f"{z['re']:.4g} {sign} {abs(z['im']):.4g}j"


def _plots_figure(dataset: Dict) -> bytes:
    t = np.array(dataset["t"])
    xs = dataset["x_star"]
    events = dataset["events"]
    fig, axes = plt.subplots(5, 1, figsize=(7.2, 10.5), sharex=True)
    series = [
        ("v_b", "Bus voltage v_b (V)", xs["v_b"]),
        ("i_l", "Line current i_l (A)", xs["i_l"]),
        ("sigma", "Integral state sigma", xs["sigma"]),
        ("v_o", "Converter voltage v_o (V)", xs["v_o"]),
    ]
    for ax, (key, label, ref) in zip(axes[:4], series):
        ax.plot(t, dataset[key], color="#1f77b4", lw=1.3)
        ax.axhline(ref, color="#888", ls="--", lw=0.8, label=f"{key}* = {ref:.4g}")
        ax.set_ylabel(label, fontsize=8)
        ax.legend(fontsize=6, loc="best")
        ax.grid(alpha=0.3)
    # residual with symlog so contraction is visible
    axr = axes[4]
    axr.plot(t, dataset["e_sigma"], color="#d62728", lw=1.3, label="e_Sigma (sim)")
    axr.plot(t, dataset["e_sigma_theory"], color="#2ca02c", ls=":", lw=1.3,
             label="e0*exp(-k_m t)")
    axr.set_yscale("symlog", linthresh=1e-6)
    axr.set_ylabel("Manifold residual e_Sigma", fontsize=8)
    axr.set_xlabel("time (s)", fontsize=8)
    axr.legend(fontsize=6, loc="best")
    axr.grid(alpha=0.3)
    for ax in axes:
        for ev in events:
            ax.axvline(ev["t"], color="#ff7f0e", ls="-.", lw=0.8, alpha=0.7)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    return buf.getvalue()


def build_pdf(dataset: Dict) -> bytes:
    prov = dataset["provenance"]
    params = prov["parameters"]
    eq = compute_equilibrium(params)
    st = compute_stability(params)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=8, leading=10))
    styles.add(ParagraphStyle(name="Mono", parent=styles["Normal"], fontName="Courier", fontSize=7.5, leading=9.5))
    h2 = styles["Heading2"]; h1 = styles["Title"]; body = styles["Normal"]; small = styles["Small"]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=16 * mm, bottomMargin=16 * mm,
                            leftMargin=16 * mm, rightMargin=16 * mm)
    E: List = []

    E.append(Paragraph("IMS Platform — Ideal Four-State Stabilizing-MRC Report", h1))
    E.append(Paragraph(f"Model: <b>{MODEL_ID}</b> v{MODEL_VERSION} &nbsp;|&nbsp; "
                       f"run_id: {prov['run_id']} &nbsp;|&nbsp; {prov['timestamp_utc']}", small))
    E.append(Spacer(1, 6))

    # Model / equations
    E.append(Paragraph("1. Model & Equations", h2))
    E.append(Paragraph(
        "State x = (i_l, v_b, sigma, v_o), domain D = {v_b &gt; 0}. Ideal continuous-time, "
        "unsaturated (dv_o/dt = u directly).", small))
    E.append(Paragraph(
        "L·di_l/dt = v_o − R·i_l − v_b &nbsp;&nbsp; C·dv_b/dt = i_l − P/v_b &nbsp;&nbsp; "
        "dσ/dt = v_nom − v_b", styles["Mono"]))
    E.append(Paragraph(
        "e_Σ = v_o − v_nom + R_v·i_l − K_i·σ &nbsp;&nbsp; "
        "u = −(R_v/L)(v_o−R·i_l−v_b) + K_i(v_nom−v_b) − k_m·e_Σ", styles["Mono"]))
    E.append(Paragraph("Exact identity (ideal model): d e_Σ/dt = −k_m·e_Σ.", styles["Mono"]))
    E.append(Spacer(1, 4))

    # Parameters
    E.append(Paragraph("2. Parameters & Units", h2))
    prows = [["Parameter", "Value", "Unit", "Description"]]
    for name, spec in PARAM_SCHEMA.items():
        prows.append([name, f"{params[name]:.6g}", spec["unit"], spec["label"]])
    pt = Table(prows, hAlign="LEFT", colWidths=[60, 80, 55, 240])
    pt.setStyle(_tbl_style())
    E.append(pt)
    E.append(Spacer(1, 4))

    # Equilibrium
    E.append(Paragraph("3. Closed-Form Equilibrium & Numerical Residual", h2))
    xs = eq["x_star"]
    E.append(Paragraph(
        f"x* = (i_l*={xs['i_l']:.6g}, v_b*={xs['v_b']:.6g}, σ*={xs['sigma']:.6g}, v_o*={xs['v_o']:.6g})", small))
    E.append(Paragraph(
        f"Numerical closed-loop equilibrium residual ||f_cl(x*)|| = "
        f"<b>{eq['equilibrium_residual_norm']:.3e}</b> &nbsp; (g = P/v_nom² = {eq['g']:.6g})", small))
    E.append(Spacer(1, 4))

    # Jacobian + stability
    E.append(Paragraph("4. Full Closed-Loop Jacobian & Local Stability", h2))
    Ja = st["jacobian_analytic"]
    jrows = [["", "i_l", "v_b", "sigma", "v_o"]]
    names = ["i_l", "v_b", "sigma", "v_o"]
    for r, nm in enumerate(names):
        jrows.append([nm] + [f"{Ja[r][c]:.4g}" for c in range(4)])
    jt = Table(jrows, hAlign="LEFT", colWidths=[45, 95, 95, 95, 95])
    jt.setStyle(_tbl_style())
    E.append(jt)
    E.append(Paragraph(
        f"Independent finite-difference Jacobian cross-check: max|J_analytic − J_numeric| = "
        f"{st['jacobian_independent_check_max_abs_error']:.2e} "
        f"({'OK' if st['jacobian_independent_check_ok'] else 'MISMATCH'}).", small))
    E.append(Spacer(1, 3))
    E.append(Paragraph(
        "Full eigenvalues (1/s): " + ", ".join(_fmt(z) for z in st["full_eigenvalues"]), small))
    E.append(Paragraph(
        "Reduced-dynamics eigenvalues (1/s): " + ", ".join(_fmt(z) for z in st["reduced_eigenvalues"]), small))
    E.append(Paragraph(
        f"Transverse eigenvalue λ⊥ = {_fmt(st['transverse_eigenvalue'])}; expected −k_m = "
        f"{st['transverse_expected']:.6g}; error = {st['transverse_check_error']:.2e} "
        f"({'OK' if st['transverse_check_ok'] else 'MISMATCH'}).", small))
    E.append(Spacer(1, 3))
    c = st["characteristic_coeffs"]
    E.append(Paragraph(
        f"Characteristic polynomial (reduced): s³ + a2 s² + a1 s + a0, "
        f"a2={c['a2']:.6g}, a1={c['a1']:.6g}, a0={c['a0']:.6g}", small))
    rrows = [["Routh-Hurwitz condition", "Satisfied", "Margin"]]
    for name, d in st["routh_hurwitz"].items():
        rrows.append([name, "yes" if d["satisfied"] else "NO", f"{d['margin']:.6g}"])
    rt = Table(rrows, hAlign="LEFT", colWidths=[180, 80, 160])
    rt.setStyle(_tbl_style())
    E.append(rt)
    E.append(Paragraph(f"<b>Local exponential stability: "
                       f"{'YES' if st['locally_exponentially_stable'] else 'NO'}.</b> "
                       + st["scope_note"], small))
    E.append(Spacer(1, 4))

    # Solver + disturbance
    E.append(Paragraph("5. Solver Settings & Disturbance", h2))
    s = prov["solver"]
    E.append(Paragraph(
        f"Solver {s['solver_name']}, rtol={s['relative_tolerance']}, atol={s['absolute_tolerance']}, "
        f"n_eval={s['n_eval']}, t_span={s['time_span']} s.", small))
    d = prov.get("disturbance")
    if d:
        E.append(Paragraph(
            f"Temporary CPL disturbance: P {d['P_nom']:.6g} W → {d['P_disturbed']:.6g} W "
            f"applied at t={d['t_start']:.6g} s, cleared at t={d['t_clear']:.6g} s.", small))
    else:
        E.append(Paragraph("No CPL disturbance configured.", small))
    E.append(Paragraph(
        f"Initial condition: {prov['initial_condition']} (perturbation {prov['initial_perturbation']}).", small))
    rc = dataset["residual_contraction"]
    E.append(Paragraph(
        f"Residual contraction vs e0·exp(−k_m t): max abs error {rc['max_abs_error']:.3e}, "
        f"max relative error {rc['max_rel_error']:.3e}.", small))
    E.append(Spacer(1, 4))

    # Plots
    E.append(Paragraph("6. Simulation Plots", h2))
    img = _plots_figure(dataset)
    E.append(Image(io.BytesIO(img), width=165 * mm, height=240 * mm))
    E.append(PageBreak())

    # Recovery + guarantee classification
    E.append(Paragraph("7. Recovery Observation (single run)", h2))
    stt = dataset["status"]
    E.append(Paragraph(
        f"solver_status={stt['solver_status']}; recovery_outcome=<b>{stt['recovery_outcome']}</b>; "
        f"termination_reason={stt['termination_reason']}; constraint_status={stt['constraint_status']}.", small))
    E.append(Paragraph(stt["disclaimer"], small))
    E.append(Spacer(1, 4))

    E.append(Paragraph("8. Assumptions, Limitations & Guarantee Classification", h2))
    for k, v in st["claim_levels"].items():
        E.append(Paragraph(f"<b>{k}</b>: {v}", small))
    E.append(Spacer(1, 3))
    E.append(Paragraph(
        "<b>Explicit statement:</b> exact residual contraction d e_Σ/dt = −k_m e_Σ belongs to the "
        "IDEAL UNSATURATED model only (no actuator bandwidth/saturation/delay/estimation). Local "
        "linear (Routh-Hurwitz) stability does NOT by itself constitute a certified regional IMS "
        "guarantee; no forward-invariant region with a Lyapunov/contraction certificate is supplied "
        "in this milestone.", small))

    doc.build(E)
    return buf.getvalue()


def _tbl_style() -> TableStyle:
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b3a55")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0b8c8")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef1f6")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ])
