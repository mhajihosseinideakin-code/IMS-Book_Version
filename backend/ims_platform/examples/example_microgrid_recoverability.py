"""
Example 2: DC Microgrid with Constant-Power Load — Bistable Recoverability
============================================================================

This case is the classic large-signal instability of converter-dominated
DC systems: a tightly-regulated constant-power load (CPL) presents a
negative incremental impedance to the bus. For a given load power there
are two equilibria -- a stable high-voltage operating point and an
unstable (saddle) low-voltage point -- and the saddle is, quite literally,
the intrinsic-manifold-based approximation of the recoverability boundary
in this simple two-state case: a disturbance landing on the low-voltage
side of it collapses the bus voltage instead of recovering.

This makes it an ideal example for showing recoverability metrics *and*
the benefit of Manifold-Reshaping Control, since a plain open-loop system
has a hard, sharply bounded recoverable region.

Run:
    python -m ims_platform.examples.example_microgrid_recoverability [output_dir]
"""
from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ims_platform.models import DCMicrogridCPL
from ims_platform.core import Simulator
from ims_platform.ims import IntrinsicManifold, RecoverabilityAnalyzer
from ims_platform.control import ScheduledLQRControl
from ims_platform.visualization import (
    plot_manifold_and_trajectory,
    plot_residual_timeseries,
    plot_recoverability_map,
    plot_recoverability_curve,
)
from ims_platform.reporting import generate_markdown_report


def main(output_dir: str = "ims_outputs_microgrid"):
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Model + Intrinsic Manifold (stable high-voltage branch)
    # ------------------------------------------------------------------
    sys_model = DCMicrogridCPL(name="dc_microgrid_cpl")
    sim = Simulator(sys_model, method="RK45")

    M = IntrinsicManifold(sys_model, param_name="P_load").build(
        alpha_range=(0.05, 1.2), n_points=60, x0_guess=np.array([0.1, 1.19]), keep_unstable=False
    )
    print(f"[Manifold] traced {len(M.points)} stable points on the high-voltage branch")

    nominal_u = np.array([0.5])
    eq = sys_model.find_equilibrium(np.array([0.44, 1.13]), u=nominal_u)
    nominal_state = eq.x_star
    print(f"[Equilibrium] x* = {nominal_state}, stable={eq.is_stable}, eigs={np.round(eq.eigenvalues, 3)}")

    # For reference: the saddle (critical boundary point) at nominal load
    saddle_candidates = sys_model.equilibria_analytic(0.5)
    print(f"[Reference] analytic equilibria at P=0.5: {saddle_candidates}")

    # ------------------------------------------------------------------
    # 2) Large voltage-dip disturbance -> open-loop trajectory
    # ------------------------------------------------------------------
    disturbed_x0 = np.array([0.68894065, 0.29546767])  # verified: diverges open loop, on saddle's low-voltage side
    traj = sim.simulate(disturbed_x0, (0.0, 3.0), u=nominal_u, n_eval=600)
    print(f"[Trajectory] open-loop success={traj.success}, final state={traj.final_state}")

    fig1 = plot_manifold_and_trajectory(
        M, traj, labels=("i (p.u.)", "v (p.u.)"),
        title="IMS: Intrinsic Manifold and Open-Loop Trajectory (DC Microgrid + CPL)",
    )
    fig1.savefig(os.path.join(output_dir, "01_manifold_trajectory.png"), dpi=160)

    fig2 = plot_residual_timeseries(traj, M, title="Manifold Residual r_m(t) — Open Loop")
    fig2.savefig(os.path.join(output_dir, "02_residual_timeseries.png"), dpi=160)

    # ------------------------------------------------------------------
    # 3) Recoverability assessment — open loop
    # ------------------------------------------------------------------
    analyzer = RecoverabilityAnalyzer(sys_model, M, sim, recovery_tol=0.1)
    report_ol = analyzer.assess(nominal_state, radius=0.9, n_samples=150, t_horizon=3.0, u=nominal_u)
    print("[Recoverability | open loop]\n" + report_ol.summary())

    fig3 = plot_recoverability_map(
        report_ol, labels=("i (p.u.)", "v (p.u.)"), title="Recoverability Map — Open Loop",
    )
    fig3.savefig(os.path.join(output_dir, "03_recoverability_map_openloop.png"), dpi=160)

    # ------------------------------------------------------------------
    # 4) Manifold-Reshaping Control (MRC) — closed loop
    # ------------------------------------------------------------------
    mrc = ScheduledLQRControl(
        sys_model, M,
        Q=np.diag([1.0, 25.0]), R=np.array([[0.5]]),
        u_min=np.array([0.05]), u_max=np.array([1.5]),
    )
    mrc.precompute_gain_schedule(u_nominal=nominal_u)
    controller = mrc.as_controller(u_nominal=nominal_u)

    traj_cl = sim.simulate(disturbed_x0, (0.0, 3.0), controller=controller, n_eval=600)
    print(f"[Trajectory] closed-loop (MRC) success={traj_cl.success}, final state={traj_cl.final_state}")

    fig4 = plot_manifold_and_trajectory(
        M, traj_cl, labels=("i (p.u.)", "v (p.u.)"),
        title="IMS + MRC: Intrinsic Manifold and Closed-Loop (MRC) Trajectory",
    )
    fig4.savefig(os.path.join(output_dir, "04_manifold_trajectory_MRC.png"), dpi=160)

    report_cl = analyzer.assess(nominal_state, radius=0.9, n_samples=150, t_horizon=3.0, controller=controller)
    print("[Recoverability | MRC closed loop]\n" + report_cl.summary())

    fig5 = plot_recoverability_map(
        report_cl, labels=("i (p.u.)", "v (p.u.)"), title="Recoverability Map — MRC Closed Loop",
    )
    fig5.savefig(os.path.join(output_dir, "05_recoverability_map_MRC.png"), dpi=160)

    # ------------------------------------------------------------------
    # 5) Recoverability curve: open loop vs. MRC, across disturbance radius
    # ------------------------------------------------------------------
    radii = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    curve_ol = analyzer.sweep_radius(nominal_state, radii, t_horizon=2.5, n_samples=30, u=nominal_u)
    curve_cl = analyzer.sweep_radius(nominal_state, radii, t_horizon=2.5, n_samples=30, controller=controller)

    fig6, ax = plt.subplots(figsize=(7, 4.5))
    plot_recoverability_curve(curve_ol, ax=ax, title="Recoverability Index vs. Disturbance Radius")
    ax.lines[0].set_label("Open loop")
    ax.plot(radii, [curve_cl[r].recoverability_index for r in radii],
            marker="s", color="#d84315", lw=2, label="MRC closed loop")
    ax.legend(fontsize=8)
    fig6.tight_layout()
    fig6.savefig(os.path.join(output_dir, "06_recoverability_curve_comparison.png"), dpi=160)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    report_md = generate_markdown_report(
        sys_model, M, report_cl,
        case_name="DC Microgrid with Constant-Power Load — Bistable Recoverability under MRC",
        controller_name="Manifold-Reshaping Control (local LQR toward nearest manifold point)",
        extra_notes=(
            f"Open-loop Recoverability Index at radius=0.9: {report_ol.recoverability_index:.2f} "
            f"vs. MRC closed-loop: {report_cl.recoverability_index:.2f} "
            f"({100*(report_cl.recoverability_index-report_ol.recoverability_index):.1f} pp improvement). "
            "This system has a genuine saddle-type unstable equilibrium (the classical CPL "
            "large-signal instability); it is the intrinsic-manifold-adjacent critical boundary "
            "separating recoverable from non-recoverable disturbances in open loop."
        ),
    )
    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(report_md)

    print(f"\nAll outputs written to: {os.path.abspath(output_dir)}")
    return dict(report_ol=report_ol, report_cl=report_cl, manifold=M)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "ims_outputs_microgrid"
    main(out)
