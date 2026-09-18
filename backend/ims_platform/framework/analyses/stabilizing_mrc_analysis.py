"""
ims_platform.framework.analyses.stabilizing_mrc_analysis
--------------------------------------------------------

The Four-State Stabilising-MRC as the first analysis fully conformant to the
shared framework. It does NOT re-implement any mathematics: it delegates to
the verified `ims_platform.stabilizing.workflow` and packages the results into
the standardized IMS result envelope.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..interfaces import AnalysisDescriptor, SupportedOutputs, build_envelope
from ...stabilizing import (
    MODEL_ID, MODEL_VERSION,
    validate_params, ParameterError,
    compute_equilibrium, compute_stability, run_simulation,
)
from ...models.stabilizing_mrc import StabilizingMRCModel


class StabilizingMRCAnalysis:
    """Standardized wrapper over the verified four-state MRC workflow."""

    analysis_id = "stabilizing_mrc"

    def descriptor(self) -> AnalysisDescriptor:
        return AnalysisDescriptor(
            id=self.analysis_id,
            title="Four-State Stabilising MRC",
            model_id=MODEL_ID,
            category="controller",
            description=(
                "Ideal continuous-time four-state stabilising manifold-regulating controller "
                "x = (i_l, v_b, sigma, v_o) with a temporary CPL power disturbance. Exact residual "
                "contraction, closed-form equilibrium, full Jacobian + Routh-Hurwitz, and observed "
                "finite-horizon recovery."
            ),
            supported_outputs=SupportedOutputs(
                equilibrium=True, manifold_residual=True, residual_trajectory=True,
                contraction_evidence=True, local_stability=True, finite_horizon_recovery=True,
                solver_constraint_status=True, report_export=True,
            ),
            guarantee_scope=(
                "Exact residual contraction is an ideal-unsaturated-model property; local exponential "
                "stability (Routh-Hurwitz) is NOT a certified regional IMS guarantee."
            ),
            standardized=True,
            entry={"type": "embedded_react", "path": "/mrc?embedded=1", "view_fn": "showStabilizingMRC"},
        )

    def default_config(self) -> Dict[str, Any]:
        params = StabilizingMRCModel().default_params()
        return {
            "params": params,
            "t_end": 0.06,
            "n_eval": 2000,
            "rtol": 1e-8,
            "atol": 1e-10,
            "method": "RK45",
            "initial_perturbation": {"i_l": 0.0, "v_b": 0.0, "sigma": 0.0, "v_o": 20.0},
            "disturbance": {"P_nom": params["P"], "P_disturbed": 1.5 * params["P"],
                            "t_start": 0.01, "t_clear": 0.03},
        }

    def validate(self, config: Dict[str, Any]) -> List[str]:
        try:
            validate_params((config or {}).get("params", {}))
            return []
        except ParameterError as e:
            return list(e.errors)

    def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        config = config or {}
        params = config.get("params", {})
        eq = compute_equilibrium(params)
        st = compute_stability(params)
        sim_cfg = {k: config[k] for k in
                   ("t_end", "n_eval", "rtol", "atol", "method", "initial_perturbation", "disturbance")
                   if k in config}
        ds = run_simulation(params, sim_cfg)

        return build_envelope(
            analysis_id=self.analysis_id,
            model_id=MODEL_ID,
            model_version=MODEL_VERSION,
            supported=self.descriptor().supported_outputs,
            equilibrium={
                "x_star": eq["x_star"],
                "state_order": eq["state_order"],
                "residual_norm": eq["equilibrium_residual_norm"],
                "g": eq["g"],
                "claim_level": "local_equilibrium_stability",
            },
            manifold={
                "residual_definition": "e_Sigma = v_o - v_nom + R_v*i_l - K_i*sigma",
                "coordinates": ["i_l", "sigma"],
                "e_sigma_0": ds["e_sigma_0"],
            },
            residual_trajectory={
                "t": ds["t"], "e_sigma": ds["e_sigma"], "e_sigma_theory": ds["e_sigma_theory"],
            },
            contraction_evidence={
                "kind": "exact_identity",
                "identity": "d e_Sigma/dt = -k_m e_Sigma",
                "rate_k_m": ds["residual_contraction"]["k_m"],
                "max_rel_error": ds["residual_contraction"]["max_rel_error"],
                "max_abs_error": ds["residual_contraction"]["max_abs_error"],
                "claim_level": "ideal_residual_contraction",
            },
            local_stability={
                "full_eigenvalues": st["full_eigenvalues"],
                "reduced_eigenvalues": st["reduced_eigenvalues"],
                "transverse_eigenvalue": st["transverse_eigenvalue"],
                "transverse_expected": st["transverse_expected"],
                "transverse_check_ok": st["transverse_check_ok"],
                "jacobian_independent_check_ok": st["jacobian_independent_check_ok"],
                "characteristic_coeffs": st["characteristic_coeffs"],
                "routh_hurwitz": st["routh_hurwitz"],
                "locally_exponentially_stable": st["locally_exponentially_stable"],
                "scope_note": st["scope_note"],
                "claim_level": "local_equilibrium_stability",
            },
            recovery_status={**ds["status"], "claim_level": "observed_finite_horizon_recovery"},
            solver_status=ds["provenance"]["solver"],
            trajectories={
                "t": ds["t"], "i_l": ds["i_l"], "v_b": ds["v_b"], "sigma": ds["sigma"],
                "v_o": ds["v_o"], "cpl_power": ds["cpl_power"],
                "disturbance_active": ds["disturbance_active"],
                "events": ds["events"], "x_star": ds["x_star"],
            },
            run_id=ds["run_id"],
            provenance=ds["provenance"],
        )
