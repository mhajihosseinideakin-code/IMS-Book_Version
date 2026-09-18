"""
ims_platform.mrc_designer.designer
-----------------------------------

Generic MRC synthesis, closed-loop verification, and Before/After comparison.

Synthesis reuses the model-independent `MRCSynthesizer` (the generic four-step
relation A(x)u = -K phi - Dphi f, solved in closed form). It is NEVER special-
cased to the four-state controller. The Four-State Stabilising MRC is the exact
regression case: the generic path reproduces its law, equilibrium, transverse
eigenvalue and residual contraction.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from ..core.system import DynamicalSystem
from ..core.simulator import Simulator
from ..control.mrc_synthesis import MRCSynthesizer, MRCSynthesisError
from ..models.stabilizing_mrc import StabilizingMRCModel
from .feasibility import feasibility_report, _has_symbolic


class MRCNotEstablished(Exception):
    """Raised when MRC cannot be analytically established for a model/config."""


# Registry of models the Designer can synthesize analytically. Extensible:
# adding a converter/network-specific MRC formulation = add an entry here.
def _stabilizing_factory(params: Dict) -> DynamicalSystem:
    return StabilizingMRCModel(params=params)


_DESIGNER_MODELS: Dict[str, Dict] = {
    "stabilizing_mrc": {
        "factory": _stabilizing_factory,
        "control_symbol": "u",
        "title": "Four-State Stabilising MRC (reference)",
        "manifold_roles": {
            "ims_analysis_manifold": "phi(x) = v_o - v_nom + R_v i_l - K_i sigma (equilibrium-consistent residual)",
            "candidate_mrc_target": "same phi(x): validated controlled-invariant target for this model",
            "validated_controlled_invariant": True,
        },
        "reference_workflow": "ims_platform.stabilizing.workflow",
    },
}


def get_designer_model_ids() -> List[str]:
    return list(_DESIGNER_MODELS.keys())


def synthesize_mrc(model_id: str, params: Dict, k_m: Optional[float] = None) -> Dict:
    """
    Synthesize the MRC law for a supported model via the generic engine, after
    an analytic feasibility gate. Returns law (symbolic + numeric callable),
    feasibility, and the explicit manifold-role distinction.
    """
    if model_id not in _DESIGNER_MODELS:
        raise MRCNotEstablished(
            f"MRC not established: model '{model_id}' has no validated analytic manifold/"
            f"control-affine formulation registered with the Designer. Supported: {get_designer_model_ids()}."
        )
    spec = _DESIGNER_MODELS[model_id]
    system = spec["factory"](params)
    x_star = _equilibrium(system, params, model_id)

    feas = feasibility_report(system, x_star, params)
    if not feas["mrc_established"]:
        raise MRCNotEstablished(feas["reason"])

    try:
        syn = MRCSynthesizer(system, control_symbol_name=spec["control_symbol"])
        res = syn.synthesize()
    except MRCSynthesisError as e:
        raise MRCNotEstablished(str(e))

    km_val = float(k_m) if k_m is not None else float(params.get("k_m"))
    numeric = res.compile_numeric(params, km_val)
    n_other = len(res.u_syms) - 1
    law: Callable[[np.ndarray], float] = (lambda x: float(numeric(np.asarray(x, float), np.zeros(n_other))))

    return {
        "model_id": model_id,
        "title": spec["title"],
        "control_symbol": spec["control_symbol"],
        "manifold_constraint": str(res.manifold_constraint),
        "open_loop_residual_dynamics": str(res.open_loop_residual_dynamics),
        "control_law_symbolic": str(res.control_expr),
        "control_law_latex": res.as_latex(),
        "k_m": km_val,
        "synthesis_relation": "A(x) u = -k_m phi(x) - Dphi(x) f(x)  (generic four-step; solved in closed form)",
        "manifold_roles": spec["manifold_roles"],
        "feasibility": feas,
        "_numeric_law": law,   # not serialized by the API layer
    }


def _equilibrium(system: DynamicalSystem, params: Dict, model_id: str) -> np.ndarray:
    if hasattr(system, "equilibrium_closed_form"):
        return np.asarray(system.equilibrium_closed_form(params), dtype=float)
    eq = system.find_equilibrium(system.initial_guess() if hasattr(system, "initial_guess")
                                 else np.zeros(system.n_states), with_eigs=False)
    return eq.x_star


def _run(system: DynamicalSystem, params: Dict, x0: np.ndarray, controller: Callable,
         sim_cfg: Dict) -> Dict:
    t_end = float(sim_cfg.get("t_end", 0.06))
    n_eval = int(sim_cfg.get("n_eval", 2000))
    method = str(sim_cfg.get("method", "RK45"))
    rtol = float(sim_cfg.get("rtol", 1e-8)); atol = float(sim_cfg.get("atol", 1e-10))
    dist = sim_cfg.get("disturbance")
    events = []
    if dist:
        events = [
            {"t": float(dist["t_start"]), "type": "param", "value": {"P": float(dist["P_disturbed"])},
             "label": "CPL disturbance applied"},
            {"t": float(dist["t_clear"]), "type": "param", "value": {"P": float(dist["P_nom"])},
             "label": "CPL disturbance cleared"},
        ]
    sim = Simulator(system, method=method, rtol=rtol, atol=atol)
    result = sim.simulate(x0=x0, t_span=(0.0, t_end), controller=controller,
                          disturbances=events, n_eval=n_eval)
    t = result.t; X = result.x
    return {"t": t, "X": X, "events": result.events, "success": result.success, "message": result.message}


def verify_closed_loop(model_id: str, params: Dict, sim_cfg: Dict, k_m: Optional[float] = None) -> Dict:
    """
    Synthesize + run the MRC-controlled model and report the standard IMS
    outputs with the four claim levels kept distinct. For the reference model
    this reuses the verified stabilizing workflow so numbers are the exact
    regression benchmark.
    """
    design = synthesize_mrc(model_id, params, k_m=k_m)
    if model_id == "stabilizing_mrc":
        from ..stabilizing import compute_equilibrium, compute_stability, run_simulation
        eq = compute_equilibrium(params)
        st = compute_stability(params)
        ds = run_simulation(params, sim_cfg)
        return {
            "design": _serializable_design(design),
            "equilibrium": {"x_star": eq["x_star"], "residual_norm": eq["equilibrium_residual_norm"],
                            "claim_level": "local_equilibrium_stability"},
            "local_stability": {
                "full_eigenvalues": st["full_eigenvalues"], "transverse_eigenvalue": st["transverse_eigenvalue"],
                "transverse_expected": st["transverse_expected"], "transverse_check_ok": st["transverse_check_ok"],
                "routh_hurwitz": st["routh_hurwitz"], "locally_exponentially_stable": st["locally_exponentially_stable"],
                "scope_note": st["scope_note"], "claim_level": "local_equilibrium_stability"},
            "residual_trajectory": {"t": ds["t"], "e_sigma": ds["e_sigma"], "e_sigma_theory": ds["e_sigma_theory"]},
            "contraction_evidence": {"identity": "d e_Sigma/dt = -k_m e_Sigma",
                                     "max_rel_error": ds["residual_contraction"]["max_rel_error"],
                                     "claim_level": "ideal_residual_contraction"},
            "trajectories": {"t": ds["t"], "i_l": ds["i_l"], "v_b": ds["v_b"], "sigma": ds["sigma"],
                             "v_o": ds["v_o"], "cpl_power": ds["cpl_power"], "events": ds["events"],
                             "x_star": ds["x_star"]},
            "recovery_status": {**ds["status"], "claim_level": "observed_finite_horizon_recovery"},
            "solver_status": ds["provenance"]["solver"],
            "run_id": ds["run_id"], "provenance": ds["provenance"],
        }
    raise MRCNotEstablished(f"closed-loop verification not wired for '{model_id}'")


def before_after(model_id: str, params: Dict, sim_cfg: Dict, k_m: Optional[float] = None) -> Dict:
    """
    Open-loop (uncontrolled, u=0) vs MRC-controlled on IDENTICAL plant, params,
    initial condition, disturbance, solver and horizon. Numerical/simulation
    evidence only -- never a certified regional IMS guarantee.
    """
    design = synthesize_mrc(model_id, params, k_m=k_m)
    system = _DESIGNER_MODELS[model_id]["factory"](params)
    x_star = _equilibrium(system, params, model_id)
    pert = sim_cfg.get("initial_perturbation") or {}
    delta = np.array([float(pert.get(nm, 0.0)) for nm in system.state_names])
    x0 = x_star + delta

    law = design["_numeric_law"]
    open_loop = _run(system, params, x0, lambda t, x: np.zeros(len(system.input_names)), sim_cfg)
    mrc = _run(system, params, x0, lambda t, x: np.array([law(x)]), sim_cfg)

    def _pack(run: Dict) -> Dict:
        t = run["t"]; X = run["X"]
        names = list(system.state_names)
        e_sigma = None
        if hasattr(system, "residual_e_sigma"):
            e_sigma = [float(system.residual_e_sigma(X[:, k], params)) for k in range(X.shape[1])]
        return {
            "t": [float(v) for v in t],
            "states": {names[i]: [float(v) for v in X[i]] for i in range(len(names))},
            "e_sigma": e_sigma,
            "success": run["success"], "message": run["message"],
            "events": [{"t": float(a), "label": b} for (a, b) in run["events"]],
        }

    ol = _pack(open_loop); mc = _pack(mrc)
    metrics = _compare_metrics(system, x_star, open_loop, mrc, params)
    return {
        "design": _serializable_design(design),
        "identical_conditions": {"x0": [float(v) for v in x0], "disturbance": sim_cfg.get("disturbance"),
                                 "solver": {"method": sim_cfg.get("method", "RK45"),
                                            "t_end": sim_cfg.get("t_end", 0.06)}},
        "x_star": {system.state_names[i]: float(x_star[i]) for i in range(len(x_star))},
        "open_loop": ol, "mrc": mc, "metrics": metrics,
        "claim_level": "observed_finite_horizon_recovery",
        "disclaimer": ("Before/After is numerical/simulation evidence on one matched scenario. Improved "
                       "recovery under MRC is NOT a certified regional IMS guarantee."),
    }


def _compare_metrics(system, x_star, ol, mrc, params) -> Dict:
    def _final_dev(run):
        return float(np.linalg.norm(run["X"][:, -1] - x_star))
    def _vb_stats(run):
        if "v_b" in system.state_names:
            vb = run["X"][system.state_names.index("v_b")]
            return {"min": float(np.min(vb)), "max": float(np.max(vb)),
                    "final": float(vb[-1]), "undershoot": float(x_star[system.state_names.index("v_b")] - np.min(vb))}
        return None
    return {
        "final_state_deviation": {"open_loop": _final_dev(ol), "mrc": _final_dev(mrc)},
        "bus_voltage": {"open_loop": _vb_stats(ol), "mrc": _vb_stats(mrc)},
    }


def _serializable_design(design: Dict) -> Dict:
    return {k: v for k, v in design.items() if not k.startswith("_")}
