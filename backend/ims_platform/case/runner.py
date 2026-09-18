"""
case.runner
-----------

`CaseRunner` is the single entry point that turns a user-authored `Case`
(system parameters + disturbance + analysis settings) into a complete,
automated IMS/MRC analysis: it builds the model, identifies the intrinsic
manifold, applies the specified disturbance, runs a recoverability
assessment (open loop, and closed loop under Manifold-Reshaping Control if
requested), generates all plots, and writes a Markdown + HTML report.

This is the layer a non-programmer-facing user interacts with: everything
they need to specify is data (a case file), not code.
"""

from __future__ import annotations

import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import Optional

from .schema import Case, CaseValidationError
from .registry import MODEL_REGISTRY, MODEL_METADATA
from ..core.simulator import Simulator
from ..ims.manifold import IntrinsicManifold
from ..ims.recoverability import RecoverabilityAnalyzer
from ..control.mrc import ScheduledLQRControl
from ..visualization.plots import (
    plot_manifold_and_trajectory,
    plot_residual_timeseries,
    plot_recoverability_map,
    plot_recoverability_curve,
)
from ..reporting.report import generate_markdown_report, generate_html_report


class CaseRunner:
    """Runs the full IMS/MRC analysis pipeline described by a `Case`."""

    def __init__(self, case: Case, verbose: bool = True):
        self.case = case
        self.verbose = verbose
        self.meta = MODEL_METADATA[case.model]

    def _log(self, msg: str):
        if self.verbose:
            print(f"[{self.case.name}] {msg}", flush=True)

    # ------------------------------------------------------------------
    def run(self) -> dict:
        t0 = time.time()
        case = self.case
        out_dir = case.output.directory
        os.makedirs(out_dir, exist_ok=True)

        # 1) Build the system model with user parameter overrides
        model_cls = MODEL_REGISTRY[case.model]
        system = model_cls(params=case.params, name=case.name)
        n = system.n_states
        self._log(f"Model '{case.model}' built ({n} states: {', '.join(system.state_names)}).")

        sim = Simulator(system, method="RK45")

        nominal_input = np.array(case.nominal_input if case.nominal_input is not None
                                  else self.meta["default_operating_input"], dtype=float)
        state_guess = np.array(case.state_guess if case.state_guess is not None
                                else self.meta["default_state_guess"], dtype=float)

        # 2) Nominal equilibrium
        eq = system.find_equilibrium(state_guess, u=nominal_input)
        if not eq.converged:
            raise RuntimeError(
                f"Could not find a nominal equilibrium from state_guess={state_guess.tolist()} "
                f"and nominal_input={nominal_input.tolist()}. Try a different state_guess in the case file."
            )
        nominal_state = eq.x_star
        self._log(f"Nominal equilibrium x* = {np.round(nominal_state, 4).tolist()} "
                   f"(stable={eq.is_stable}, eigenvalues={np.round(eq.eigenvalues, 3).tolist()})")

        # 3) Intrinsic manifold
        sweep_range = case.manifold.sweep_param_range or self.meta["default_manifold_range"]
        M = IntrinsicManifold(system, param_name=case.model + "_operating_param").build(
            alpha_range=tuple(sweep_range),
            n_points=case.manifold.sweep_points,
            x0_guess=state_guess,
            keep_unstable=case.manifold.keep_unstable,
        )
        n_stable = sum(p.is_stable for p in M.points)
        self._log(f"Intrinsic manifold traced: {len(M.points)} points "
                   f"({n_stable} stable / {len(M.points) - n_stable} unstable).")
        if len(M.points) == 0:
            raise RuntimeError(
                "Manifold continuation produced zero points. Check manifold.sweep_param_range "
                "and state_guess in the case file."
            )

        results = dict(system=system, manifold=M, nominal_state=nominal_state, equilibrium=eq)
        can_plot_2d = (n == 2)
        if not can_plot_2d:
            self._log(f"Note: phase-plane plots are skipped for {n}-state models "
                       f"(current visualization module supports 2-state models); "
                       f"numeric recoverability metrics are unaffected.")

        # 4) Disturbance + open-loop trajectory
        dist = case.disturbance
        if dist.type == "offset":
            disturbed_x0 = nominal_state + np.array(dist.value, dtype=float)
        else:
            disturbed_x0 = np.array(dist.value, dtype=float)

        traj_ol = sim.simulate(disturbed_x0, (0.0, dist.t_horizon), u=nominal_input, n_eval=500)
        self._log(f"Open-loop disturbance trajectory simulated (success={traj_ol.success}); "
                   f"final state = {np.round(traj_ol.final_state, 4).tolist()}.")
        results["trajectory_open_loop"] = traj_ol

        figs = {}
        if can_plot_2d:
            figs["01_manifold_trajectory"] = plot_manifold_and_trajectory(
                M, traj_ol, labels=self._state_labels(system),
                title=f"IMS: Intrinsic Manifold and Open-Loop Trajectory — {case.name}",
            )
            figs["02_residual_timeseries"] = plot_residual_timeseries(
                traj_ol, M, title=f"Manifold Residual r_m(t) — Open Loop — {case.name}"
            )

        # 5) Recoverability assessment (open loop)
        report_ol = None
        if case.recoverability.enabled:
            analyzer = RecoverabilityAnalyzer(system, M, sim, recovery_tol=case.recoverability.recovery_tol)
            report_ol = analyzer.assess(
                nominal_state, radius=case.recoverability.radius,
                n_samples=case.recoverability.n_samples, t_horizon=case.recoverability.t_horizon,
                u=nominal_input,
            )
            self._log(f"Recoverability (open loop): {report_ol.summary().replace(chr(10), ' | ')}")
            results["recoverability_open_loop"] = report_ol
            if can_plot_2d:
                figs["03_recoverability_map_openloop"] = plot_recoverability_map(
                    report_ol, labels=self._state_labels(system), title=f"Recoverability Map — Open Loop — {case.name}"
                )

        # 6) Optional Manifold-Reshaping Control (closed loop)
        report_cl = None
        controller_name = None
        if case.control.enabled:
            Q = np.diag(case.control.Q_diag) if case.control.Q_diag else np.eye(n)
            R = np.diag(case.control.R_diag) if case.control.R_diag else 0.1 * np.eye(max(len(nominal_input), 1))
            u_min = np.array(case.control.u_min) if case.control.u_min else None
            u_max = np.array(case.control.u_max) if case.control.u_max else None

            mrc = ScheduledLQRControl(system, M, Q=Q, R=R, u_min=u_min, u_max=u_max)
            mrc.precompute_gain_schedule(u_nominal=nominal_input)
            controller = mrc.as_controller(u_nominal=nominal_input)
            controller_name = "Manifold-Reshaping Control (local LQR gain schedule over the intrinsic manifold)"
            self._log("MRC gain schedule precomputed; running closed-loop analysis.")

            traj_cl = sim.simulate(disturbed_x0, (0.0, dist.t_horizon), controller=controller, n_eval=500)
            self._log(f"Closed-loop (MRC) trajectory simulated (success={traj_cl.success}); "
                       f"final state = {np.round(traj_cl.final_state, 4).tolist()}.")
            results["trajectory_closed_loop"] = traj_cl
            results["mrc"] = mrc

            if can_plot_2d:
                figs["04_manifold_trajectory_MRC"] = plot_manifold_and_trajectory(
                    M, traj_cl, labels=self._state_labels(system),
                    title=f"IMS + MRC: Closed-Loop Trajectory — {case.name}",
                )

            if case.recoverability.enabled:
                report_cl = analyzer.assess(
                    nominal_state, radius=case.recoverability.radius,
                    n_samples=case.recoverability.n_samples, t_horizon=case.recoverability.t_horizon,
                    controller=controller,
                )
                self._log(f"Recoverability (MRC closed loop): {report_cl.summary().replace(chr(10), ' | ')}")
                results["recoverability_closed_loop"] = report_cl
                if can_plot_2d:
                    figs["05_recoverability_map_MRC"] = plot_recoverability_map(
                        report_cl, labels=self._state_labels(system), title=f"Recoverability Map — MRC Closed Loop — {case.name}"
                    )

            # radius sweep comparison curve
            if case.recoverability.enabled and case.recoverability.radius_sweep:
                radii = case.recoverability.radius_sweep
                curve_ol = analyzer.sweep_radius(nominal_state, radii, t_horizon=case.recoverability.t_horizon,
                                                  n_samples=case.recoverability.radius_sweep_samples, u=nominal_input)
                curve_cl = analyzer.sweep_radius(nominal_state, radii, t_horizon=case.recoverability.t_horizon,
                                                  n_samples=case.recoverability.radius_sweep_samples, controller=controller)
                fig, ax = plt.subplots(figsize=(7, 4.5))
                plot_recoverability_curve(curve_ol, ax=ax, title=f"Recoverability Index vs. Disturbance Radius — {case.name}")
                ax.lines[0].set_label("Open loop")
                ax.plot(radii, [curve_cl[r].recoverability_index for r in radii],
                        marker="s", color="#d84315", lw=2, label="MRC closed loop")
                ax.legend(fontsize=8)
                fig.tight_layout()
                figs["06_recoverability_curve_comparison"] = fig
                results["recoverability_curve_open_loop"] = curve_ol
                results["recoverability_curve_closed_loop"] = curve_cl

        # 7) Save plots
        if "png" in case.output.formats:
            for fname, fig in figs.items():
                fig.savefig(os.path.join(out_dir, f"{fname}.png"), dpi=160)
            self._log(f"Saved {len(figs)} plot(s) to {out_dir}/")
        for fig in figs.values():
            plt.close(fig)

        # 8) Reports
        report_for_summary = report_cl if report_cl is not None else report_ol
        if report_for_summary is not None:
            if "md" in case.output.formats:
                md = generate_markdown_report(
                    system, M, report_for_summary, case_name=case.name,
                    controller_name=controller_name,
                    extra_notes=self._build_extra_notes(report_ol, report_cl),
                )
                with open(os.path.join(out_dir, "report.md"), "w") as f:
                    f.write(md)
            if "html" in case.output.formats:
                html = generate_html_report(
                    system, M, report_for_summary, case_name=case.name,
                    controller_name=controller_name,
                    extra_notes=self._build_extra_notes(report_ol, report_cl),
                    image_dir=out_dir,
                    image_names=[f"{k}.png" for k in figs.keys()] if "png" in case.output.formats else [],
                )
                with open(os.path.join(out_dir, "report.html"), "w") as f:
                    f.write(html)
            self._log(f"Report(s) written to {out_dir}/")
        else:
            self._log("Recoverability assessment disabled; skipping report generation "
                       "(equilibrium/manifold results are still available programmatically).")

        # 9) Persist the case file itself alongside outputs for reproducibility
        with open(os.path.join(out_dir, "case_used.json"), "w") as f:
            json.dump(case.to_dict(), f, indent=2, default=str)

        elapsed = time.time() - t0
        self._log(f"Done in {elapsed:.1f}s. All outputs in: {os.path.abspath(out_dir)}")
        results["output_dir"] = os.path.abspath(out_dir)
        results["elapsed_seconds"] = elapsed
        return results

    @staticmethod
    def _state_labels(system):
        return tuple(system.state_names) if system.state_names else (None, None)

    @staticmethod
    def _build_extra_notes(report_ol, report_cl) -> Optional[str]:
        if report_ol is None:
            return None
        if report_cl is None:
            return None
        delta = report_cl.recoverability_index - report_ol.recoverability_index
        return (
            f"Open-loop Recoverability Index: {report_ol.recoverability_index:.2f} vs. "
            f"MRC closed-loop: {report_cl.recoverability_index:.2f} "
            f"({100*delta:+.1f} percentage points)."
        )
