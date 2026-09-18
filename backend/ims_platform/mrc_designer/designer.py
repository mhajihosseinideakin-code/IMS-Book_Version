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
from ..models.converter_cpl_paper import ConverterCPLPaper
from .feasibility import feasibility_report, _has_symbolic

#: Nominal DC bus voltage used to anchor the converter-CPL equilibrium family
#: (the outer-loop state v_o is free at u=0, so the equilibrium is a 1-parameter
#: family; the assembled Network + Auto-MRC demo anchors it at the nominal bus).
_CPL_V_NOM = 400.0


class MRCNotEstablished(Exception):
    """Raised when MRC cannot be analytically established for a model/config."""


# Registry of models the Designer can synthesize analytically. Extensible:
# adding a converter/network-specific MRC formulation = add an entry here.
def _stabilizing_factory(params: Dict) -> DynamicalSystem:
    return StabilizingMRCModel(params=params)


def _converter_cpl_factory(params: Dict) -> DynamicalSystem:
    return ConverterCPLPaper(params=params)


def _converter_cpl_equilibrium(p: Dict) -> np.ndarray:
    # Anchored at the nominal bus V: di_l=0 -> v_o = R i_l + v_b; dv_b=0 ->
    # i_l = P/v_b; dv_o = u = 0. (i_l, v_b, v_o) order of ConverterCPLPaper.
    V = _CPL_V_NOM
    i_l = float(p["P"]) / V
    return np.array([i_l, V, V + float(p["R"]) * i_l])


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
    # The assembled Network + Auto-MRC demo: its MRC law is derived by the
    # Symbolic Engine from ConverterCPLPaper (paper eq. 1-6/13). Registered here
    # so a project's downstream "Design MRC Controller" stage synthesizes on the
    # SAME validated model the project's IMS analysis used -- no substitution of
    # the four-state reference.
    "converter_cpl_paper": {
        "factory": _converter_cpl_factory,
        "control_symbol": "u",
        "title": "Assembled Converter-CPL Network (Auto-MRC)",
        "manifold_roles": {
            "ims_analysis_manifold": "e_m(x) = v_b - (v_o - R i_l)  (paper eq. 6, on the assembled network)",
            "candidate_mrc_target": "same e_m(x): analytic controlled-invariant target derived by the Symbolic Engine",
            "validated_controlled_invariant": True,
        },
        "equilibrium": _converter_cpl_equilibrium,
        "full_stability_note": (
            "This assembled reconstruction has a documented, structurally-positive tangential mode "
            "(P/(C v_b^2), independent of parameters; see tests/test_mrc_synthesis.py). MRC guarantees "
            "the manifold residual's exponential contraction at rate k_m (transverse mode = -k_m), NOT "
            "full local asymptotic stability of this specific 3-state fixture."
        ),
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
    spec = _DESIGNER_MODELS.get(model_id, {})
    if callable(spec.get("equilibrium")):
        return np.asarray(spec["equilibrium"](params), dtype=float)
    if hasattr(system, "equilibrium_closed_form"):
        return np.asarray(system.equilibrium_closed_form(params), dtype=float)
    eq = system.find_equilibrium(system.initial_guess() if hasattr(system, "initial_guess")
                                 else np.zeros(system.n_states), with_eigs=False)
    return eq.x_star


def feasibility_for(model_id: str, params: Dict) -> Dict:
    """
    Feasibility report for a registered designer model at its equilibrium.
    Raises MRCNotEstablished if the model is not registered (so the API layer
    can report an explicit, honest 'not established' instead of a hard 404).
    """
    if model_id not in _DESIGNER_MODELS:
        raise MRCNotEstablished(
            f"MRC not established: model '{model_id}' has no validated analytic manifold / "
            f"control-affine (f, G) formulation registered with the Designer. Analytic MRC synthesis "
            f"requires a symbolic manifold phi(x) and a control-affine split; only diagnostic feasibility "
            f"would be possible for this assembled model. Supported: {get_designer_model_ids()}."
        )
    system = _DESIGNER_MODELS[model_id]["factory"](params)
    x_star = _equilibrium(system, params, model_id)
    return feasibility_report(system, x_star, params)


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
    # ---- generic path (e.g. assembled converter_cpl_paper network) ----------
    spec = _DESIGNER_MODELS[model_id]
    system = spec["factory"](params)
    x_star = _equilibrium(system, params, model_id)
    km_val = float(design["k_m"])
    law = design["_numeric_law"]
    p = params
    n = system.n_states

    def _cl_field(x: np.ndarray) -> np.ndarray:
        u = np.array([law(x)])
        return np.asarray(system.dynamics(0.0, x, u, p), dtype=float)

    # closed-loop equilibrium residual (numerically verified, not just the formula)
    res_norm = float(np.linalg.norm(_cl_field(x_star)))

    # closed-loop Jacobian via finite difference (route independent of synthesis)
    J = np.zeros((n, n)); h = 1e-6; f0 = _cl_field(x_star)
    for j in range(n):
        xp = x_star.astype(float).copy(); xp[j] += h
        J[:, j] = (_cl_field(xp) - f0) / h
    eigs = np.linalg.eigvals(J)
    idx = int(np.argmin(np.abs(eigs - (-km_val))))
    lam_perp = eigs[idx]
    transverse_ok = bool(abs(lam_perp.real + km_val) < max(1e-3 * km_val, 1.0) and abs(lam_perp.imag) < 1.0)
    locally_stable = bool(np.all(eigs.real < 0))

    # residual contraction from a deliberately off-manifold initial condition
    pert = sim_cfg.get("initial_perturbation") or {}
    delta = np.array([float(pert.get(nm, 0.0)) for nm in system.state_names])
    x0 = x_star + delta
    run = _run(system, params, x0, lambda t, x: np.array([law(x)]), sim_cfg)
    t = run["t"]; X = run["X"]
    resid_fn = getattr(system, "residual_e_sigma", None) or getattr(system, "manifold_residual_numeric", None)
    e = np.array([float(resid_fn(X[:, k], params)) for k in range(X.shape[1])])
    e0 = float(e[0])
    e_theory = e0 * np.exp(-km_val * np.asarray(t))
    denom = max(abs(e0), 1e-12)
    max_rel = float(np.max(np.abs(e - e_theory)) / denom)

    scope_note = spec.get("full_stability_note", "") or (
        "Local linear stability is a linearisation claim, distinct from a certified regional IMS guarantee.")
    return {
        "design": _serializable_design(design),
        "equilibrium": {"x_star": {system.state_names[i]: float(x_star[i]) for i in range(n)},
                        "residual_norm": res_norm, "claim_level": "local_equilibrium_stability"},
        "local_stability": {
            "full_eigenvalues": [{"re": float(z.real), "im": float(z.imag)} for z in eigs],
            "transverse_eigenvalue": {"re": float(lam_perp.real), "im": float(lam_perp.imag)},
            "transverse_expected": -km_val, "transverse_check_ok": transverse_ok,
            "routh_hurwitz": [], "locally_exponentially_stable": locally_stable,
            "scope_note": scope_note, "claim_level": "local_equilibrium_stability"},
        "residual_trajectory": {"t": [float(v) for v in t], "e_sigma": [float(v) for v in e],
                                "e_sigma_theory": [float(v) for v in e_theory]},
        "contraction_evidence": {"identity": "d e_m/dt = -k_m e_m",
                                 "max_rel_error": max_rel, "claim_level": "ideal_residual_contraction"},
        "trajectories": {"t": [float(v) for v in t],
                         "states": {system.state_names[i]: [float(v) for v in X[i]] for i in range(n)},
                         "events": [{"t": float(a), "label": b} for (a, b) in run["events"]],
                         "x_star": {system.state_names[i]: float(x_star[i]) for i in range(n)}},
        "recovery_status": {"recovery_outcome": ("observed_recovery" if run["success"] and abs(e[-1]) < denom else "observed_no_recovery"),
                            "solver_success": run["success"], "claim_level": "observed_finite_horizon_recovery"},
        "solver_status": {"method": sim_cfg.get("method", "RK45"), "success": run["success"], "message": run["message"]},
        "run_id": None,
    }


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
        resid_fn = getattr(system, "residual_e_sigma", None) or getattr(system, "manifold_residual_numeric", None)
        if resid_fn is not None:
            e_sigma = [float(resid_fn(X[:, k], params)) for k in range(X.shape[1])]
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
