"""
Example 1: Grid-Forming Inverter — Intrinsic Manifold, Recoverability, and MRC
================================================================================

End-to-end demonstration:
    1. Build the model and identify its intrinsic manifold M by sweeping
       the active-power setpoint.
    2. Simulate a large disturbance (a phase-angle jump, e.g. a fault
       clearing event) in OPEN LOOP and examine the manifold residual.
    3. Run a Monte-Carlo recoverability assessment around the nominal
       operating point (open loop).
    4. Design Manifold-Reshaping Control (MRC) and repeat the
       recoverability assessment in CLOSED LOOP to show the enlargement
       of the recoverable region.
    5. Sweep disturbance radius to build a recoverability curve for both
       cases, and emit a Markdown report.

Run:
    python -m ims_platform.examples.example_inverter_analysis [output_dir]
"""
from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ims_platform.models import GridFormingInverter
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


def main(output_dir: str = "ims_outputs_inverter"):
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Model + Intrinsic Manifold
    # ------------------------------------------------------------------
    sys_model = GridFormingInverter(name="grid_forming_inverter")
    sim = Simulator(sys_model)

    M = IntrinsicManifold(sys_model, param_name="P_set").build(
        alpha_range=(0.0, 0.95), n_points=80, x0_guess=np.array([0.3, 0.0])
    )
    print(f"[Manifold] traced {len(M.points)} points "
          f"({sum(p.is_stable for p in M.points)} stable / "
          f"{sum(not p.is_stable for p in M.points)} unstable)")

    nominal_u = np.array([0.5])
    eq = sys_model.find_equilibrium(np.array([0.3, 0.0]), u=nominal_u)
    nominal_state = eq.x_star
    print(f"[Equilibrium] x* = {nominal_state}, stable={eq.is_stable}, "
          f"eigs={np.round(eq.eigenvalues, 3)}")

    # ------------------------------------------------------------------
    # 2) Large-disturbance open-loop trajectory (e.g. fault-induced angle jump)
    # ------------------------------------------------------------------
    disturbed_x0 = nominal_state + np.array([1.4, 3.0])  # large angle+speed disturbance
    traj = sim.simulate(disturbed_x0, (0.0, 12.0), u=nominal_u, n_eval=600)
    print(f"[Trajectory] open-loop success={traj.success}, final state={traj.final_state}")

    fig1 = plot_manifold_and_trajectory(
        M, traj, labels=("delta (rad)", "omega (rad/s)"),
        title="IMS: Intrinsic Manifold and Open-Loop Trajectory (Grid-Forming Inverter)",
    )
    fig1.savefig(os.path.join(output_dir, "01_manifold_trajectory.png"), dpi=160)

    fig2 = plot_residual_timeseries(traj, M, title="Manifold Residual r_m(t) — Open Loop")
    fig2.savefig(os.path.join(output_dir, "02_residual_timeseries.png"), dpi=160)

    # ------------------------------------------------------------------
    # 3) Recoverability assessment — open loop
    # ------------------------------------------------------------------
    analyzer = RecoverabilityAnalyzer(sys_model, M, sim, recovery_tol=0.15)
    report_ol = analyzer.assess(nominal_state, radius=3.0, n_samples=100, t_horizon=10, u=nominal_u)
    print("[Recoverability | open loop]\n" + report_ol.summary())

    fig3 = plot_recoverability_map(
        report_ol, labels=("delta (rad)", "omega (rad/s)"),
        title="Recoverability Map — Open Loop",
    )
    fig3.savefig(os.path.join(output_dir, "03_recoverability_map_openloop.png"), dpi=160)

    # ------------------------------------------------------------------
    # 4) Manifold-Reshaping Control (MRC) — closed loop
    # ------------------------------------------------------------------
    mrc = ScheduledLQRControl(
        sys_model, M,
        Q=np.diag([8.0, 2.0]), R=np.array([[0.05]]),
        u_min=np.array([-0.2]), u_max=np.array([1.2]),
        gain_refresh_every=40,
    )
    controller = mrc.as_controller(u_nominal=nominal_u)

    traj_cl = sim.simulate(disturbed_x0, (0.0, 12.0), controller=controller, n_eval=600)
    print(f"[Trajectory] closed-loop (MRC) success={traj_cl.success}, final state={traj_cl.final_state}")

    fig4 = plot_manifold_and_trajectory(
        M, traj_cl, labels=("delta (rad)", "omega (rad/s)"),
        title="IMS + MRC: Intrinsic Manifold and Closed-Loop (MRC) Trajectory",
    )
    fig4.savefig(os.path.join(output_dir, "04_manifold_trajectory_MRC.png"), dpi=160)

    report_cl = analyzer.assess(nominal_state, radius=3.0, n_samples=100, t_horizon=10, controller=controller)
    print("[Recoverability | MRC closed loop]\n" + report_cl.summary())

    fig5 = plot_recoverability_map(
        report_cl, labels=("delta (rad)", "omega (rad/s)"),
        title="Recoverability Map — MRC Closed Loop",
    )
    fig5.savefig(os.path.join(output_dir, "05_recoverability_map_MRC.png"), dpi=160)

    # ------------------------------------------------------------------
    # 5) Recoverability curve: open loop vs. MRC, across disturbance radius
    # ------------------------------------------------------------------
    radii = [0.5, 1.0, 1.5, 2.0, 3.0]
    curve_ol = analyzer.sweep_radius(nominal_state, radii, t_horizon=8, n_samples=25, u=nominal_u)
    curve_cl = analyzer.sweep_radius(nominal_state, radii, t_horizon=8, n_samples=25, controller=controller)

    fig6, ax = plt.subplots(figsize=(7, 4.5))
    plot_recoverability_curve(curve_ol, ax=ax, title="Recoverability Index vs. Disturbance Radius")
    ax.lines[0].set_label("Open loop")
    ax.plot([r for r in radii], [curve_cl[r].recoverability_index for r in radii],
            marker="s", color="#d84315", lw=2, label="MRC closed loop")
    ax.legend(fontsize=8)
    fig6.tight_layout()
    fig6.savefig(os.path.join(output_dir, "06_recoverability_curve_comparison.png"), dpi=160)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    report_md = generate_markdown_report(
        sys_model, M, report_cl,
        case_name="Grid-Forming Inverter — Large-Signal Recoverability under MRC",
        controller_name="Manifold-Reshaping Control (local LQR toward nearest manifold point)",
        extra_notes=(
            f"Open-loop Recoverability Index at radius=3.0: {report_ol.recoverability_index:.2f} "
            f"vs. MRC closed-loop: {report_cl.recoverability_index:.2f} "
            f"({100*(report_cl.recoverability_index-report_ol.recoverability_index):.1f} pp improvement)."
        ),
    )
    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(report_md)

    print(f"\nAll outputs written to: {os.path.abspath(output_dir)}")
    return dict(report_ol=report_ol, report_cl=report_cl, manifold=M)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "ims_outputs_inverter"
    main(out)
