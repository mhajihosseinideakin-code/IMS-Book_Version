"""
ims_platform.stabilizing.workflow
----------------------------------

Orchestration for the ideal four-state stabilizing-MRC vertical slice.
Produces ONE verified simulation dataset that is reused for UI plots,
CSV export and the PDF report.
"""

from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional

import numpy as np

from ..core.simulator import Simulator
from .model import StabilizingMRCClosedLoop

MODEL_ID = "ideal_four_state_stabilizing_mrc"
MODEL_VERSION = "1.0.0"

# ----------------------------------------------------------------------
# Parameter schema (name -> (unit, admissibility rule, human description))
# ----------------------------------------------------------------------
PLANT_PARAMS = ("R", "L", "C", "P", "v_nom")
CONTROLLER_PARAMS = ("R_v", "K_i", "k_m")

PARAM_SCHEMA: Dict[str, Dict] = {
    "R":     {"unit": "ohm",   "rule": ">=0", "group": "plant",      "label": "Line resistance"},
    "L":     {"unit": "H",     "rule": ">0",  "group": "plant",      "label": "Line inductance"},
    "C":     {"unit": "F",     "rule": ">0",  "group": "plant",      "label": "Bus capacitance"},
    "P":     {"unit": "W",     "rule": ">=0", "group": "cpl",        "label": "Nominal CPL power"},
    "v_nom": {"unit": "V",     "rule": ">0",  "group": "plant",      "label": "Nominal bus voltage"},
    "R_v":   {"unit": "ohm",   "rule": ">0",  "group": "controller", "label": "Virtual droop resistance R_v"},
    "K_i":   {"unit": "1/s",   "rule": ">0",  "group": "controller", "label": "Integral gain K_i"},
    "k_m":   {"unit": "1/s",   "rule": ">0",  "group": "controller", "label": "Manifold contraction rate k_m"},
}


class ParameterError(ValueError):
    """Raised when a configuration is physically/mathematically inadmissible."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def _check_rule(name: str, value: float, rule: str) -> Optional[str]:
    if not isinstance(value, (int, float)) or not np.isfinite(value):
        return f"{name} must be a finite number (got {value!r})"
    if rule == ">0" and not value > 0:
        return f"{name} must be > 0 ({name}={value})"
    if rule == ">=0" and not value >= 0:
        return f"{name} must be >= 0 ({name}={value})"
    return None


def validate_params(params: Dict) -> Dict:
    """
    Validate physical/mathematical admissibility of the plant + CPL +
    controller parameters. Raises ParameterError (explicit) rather than
    letting an inadmissible config silently produce numbers.

    Returns the normalized (float-coerced) parameter dict on success.
    """
    errors: List[str] = []
    normalized: Dict[str, float] = {}
    for name, spec in PARAM_SCHEMA.items():
        if name not in params or params[name] is None:
            errors.append(f"missing required parameter '{name}' ({spec['label']}, {spec['unit']})")
            continue
        try:
            val = float(params[name])
        except (TypeError, ValueError):
            errors.append(f"{name} must be numeric (got {params[name]!r})")
            continue
        err = _check_rule(name, val, spec["rule"])
        if err:
            errors.append(err)
        else:
            normalized[name] = val

    if errors:
        raise ParameterError(errors)

    # Domain admissibility of the equilibrium itself: v_b* = v_nom > 0 already
    # enforced; equilibrium must lie in D = {v_b > 0}. Nothing further to reject
    # here (RH stability is reported separately, an unstable-but-admissible
    # config is a valid simulation input, not an invalid parameter set).
    return normalized


def validate_disturbance(dist: Dict, t_end: float) -> Dict:
    """Validate the temporary CPL disturbance definition."""
    errors: List[str] = []
    out: Dict[str, float] = {}
    for key in ("P_nom", "P_disturbed", "t_start", "t_clear"):
        if key not in dist or dist[key] is None:
            errors.append(f"missing disturbance field '{key}'")
            continue
        try:
            out[key] = float(dist[key])
        except (TypeError, ValueError):
            errors.append(f"disturbance '{key}' must be numeric (got {dist[key]!r})")
    if errors:
        raise ParameterError(errors)

    if out["P_nom"] < 0 or out["P_disturbed"] < 0:
        errors.append("CPL powers must be >= 0")
    if not (0.0 <= out["t_start"] < out["t_clear"] <= t_end):
        errors.append(
            f"require 0 <= t_start ({out['t_start']}) < t_clear ({out['t_clear']}) <= t_end ({t_end})"
        )
    if errors:
        raise ParameterError(errors)
    return out


# ----------------------------------------------------------------------
# Equilibrium (closed form + numerical residual)
# ----------------------------------------------------------------------
def compute_equilibrium(params: Dict) -> Dict:
    p = validate_params(params)
    model = StabilizingMRCClosedLoop(params=p)
    x_star = model.equilibrium_closed_form(p)
    # Numerically substitute x* into the COMPLETE closed-loop field.
    f_cl = model.rhs(0.0, x_star)  # dynamics() uses the control law internally
    residual_norm = float(np.linalg.norm(f_cl))
    g = p["P"] / p["v_nom"] ** 2
    return {
        "x_star": {
            "i_l": float(x_star[0]), "v_b": float(x_star[1]),
            "sigma": float(x_star[2]), "v_o": float(x_star[3]),
        },
        "x_star_vector": [float(v) for v in x_star],
        "state_order": ["i_l", "v_b", "sigma", "v_o"],
        "f_cl_at_x_star": [float(v) for v in f_cl],
        "equilibrium_residual_norm": residual_norm,
        "g": float(g),
        "formulas": {
            "v_b_star": "v_nom",
            "i_l_star": "P / v_nom",
            "sigma_star": "(R + R_v) * P / (K_i * v_nom)",
            "v_o_star": "v_nom + R * P / v_nom",
        },
    }


# ----------------------------------------------------------------------
# Jacobian + local stability
# ----------------------------------------------------------------------
def compute_stability(params: Dict) -> Dict:
    p = validate_params(params)
    model = StabilizingMRCClosedLoop(params=p)
    x_star = model.equilibrium_closed_form(p)

    # Displayed analytic full 4x4 closed-loop Jacobian.
    J_analytic = model.analytic_closed_loop_jacobian(x_star, p)
    # INDEPENDENT route: central-difference Jacobian of the closed-loop field.
    J_numeric = model.jacobian(x_star)
    jac_match_error = float(np.max(np.abs(J_analytic - J_numeric)))

    full_eigs = np.linalg.eigvals(J_analytic)
    # Sort by real part (desc) for stable presentation.
    order = np.argsort(-full_eigs.real)
    full_eigs = full_eigs[order]

    # Reduced (i_l, v_b, sigma) eigenvalues from the cubic char. poly.
    reduced_eigs = model.poles(p)
    reduced_order = np.argsort(-reduced_eigs.real)
    reduced_eigs = reduced_eigs[reduced_order]

    # Transverse mode: block-triangular spectrum => -k_m is an eigenvalue of J.
    k_m = p["k_m"]
    # Identify the full eigenvalue closest to -k_m as the transverse mode.
    idx_perp = int(np.argmin(np.abs(full_eigs - (-k_m))))
    lambda_perp = full_eigs[idx_perp]
    lambda_perp_error = float(abs(lambda_perp - (-k_m)))

    rh = model.routh_hurwitz_report(p)

    def _c(z):
        return {"re": float(np.real(z)), "im": float(np.imag(z))}

    return {
        "jacobian_analytic": [[float(v) for v in row] for row in J_analytic],
        "jacobian_numeric": [[float(v) for v in row] for row in J_numeric],
        "jacobian_independent_check_max_abs_error": jac_match_error,
        "jacobian_independent_check_ok": bool(jac_match_error < 1e-4),
        "full_eigenvalues": [_c(z) for z in full_eigs],
        "reduced_eigenvalues": [_c(z) for z in reduced_eigs],
        "transverse_eigenvalue": _c(lambda_perp),
        "transverse_expected": -float(k_m),
        "transverse_check_error": lambda_perp_error,
        "transverse_check_ok": bool(lambda_perp_error < 1e-6 * max(1.0, k_m)),
        "characteristic_coeffs": {"a2": rh["a2"], "a1": rh["a1"], "a0": rh["a0"]},
        "routh_hurwitz": rh["conditions"],
        "locally_exponentially_stable": rh["locally_exponentially_stable"],
        "scope_note": rh["scope_note"],
        "claim_levels": _CLAIM_LEVELS,
    }


_CLAIM_LEVELS = {
    "ideal_residual_contraction": (
        "EXACT (ideal unsaturated model): de_Sigma/dt = -k_m e_Sigma while the "
        "trajectory remains in D = {v_b > 0}. Holds for the declared ideal model only."
    ),
    "local_equilibrium_stability": (
        "LOCAL ANALYTICAL: under the Routh-Hurwitz inequalities the isolated "
        "equilibrium is locally exponentially stable (linearisation). This is NOT a "
        "certified region of attraction."
    ),
    "observed_finite_horizon_recovery": (
        "NUMERICAL / EMPIRICAL: a single finite-horizon trajectory returning near x* "
        "is observed evidence only; it does not certify a basin."
    ),
    "certified_regional_ims": (
        "NOT ESTABLISHED in this milestone: no explicit forward-invariant region with a "
        "Lyapunov/contraction certificate for the reduced dynamics is supplied."
    ),
}


# ----------------------------------------------------------------------
# Simulation (closed loop + temporary CPL disturbance) -> ONE dataset
# ----------------------------------------------------------------------
def run_simulation(params: Dict, sim_cfg: Dict) -> Dict:
    """
    Run the closed-loop ideal four-state model with a temporary CPL power
    disturbance applied as a time-dependent PARAMETER change.

    sim_cfg keys:
        t_end (s), n_eval, rtol, atol, method,
        initial_perturbation: {i_l, v_b, sigma, v_o} deltas added to x*,
        disturbance: {P_nom, P_disturbed, t_start, t_clear},
        recovery: {dwell_frac, eps_r, eps_x, v_min, M_div}
    """
    p = validate_params(params)
    t_end = float(sim_cfg.get("t_end", 0.1))
    if not t_end > 0:
        raise ParameterError([f"t_end must be > 0 (got {t_end})"])
    n_eval = int(sim_cfg.get("n_eval", 2000))
    rtol = float(sim_cfg.get("rtol", 1e-8))
    atol = float(sim_cfg.get("atol", 1e-10))
    method = str(sim_cfg.get("method", "RK45"))

    dist_cfg = sim_cfg.get("disturbance") or {}
    disturbance = validate_disturbance(dist_cfg, t_end) if dist_cfg else None

    # CPL nominal power for the model: from disturbance P_nom if given else params P.
    P_nom = disturbance["P_nom"] if disturbance else p["P"]
    p_run = dict(p, P=P_nom)

    model = StabilizingMRCClosedLoop(params=p_run)
    x_star = model.equilibrium_closed_form(p_run)

    pert = sim_cfg.get("initial_perturbation") or {}
    delta = np.array([
        float(pert.get("i_l", 0.0)), float(pert.get("v_b", 0.0)),
        float(pert.get("sigma", 0.0)), float(pert.get("v_o", 0.0)),
    ])
    x0 = x_star + delta

    if not model.admissible(x0):
        raise ParameterError([f"initial condition inadmissible: v_b0={x0[1]} must be > 0"])

    controller = lambda t, x: np.array([model.closed_loop_control(x, model.params)])

    dist_events = []
    if disturbance:
        dist_events = [
            {"t": disturbance["t_start"], "type": "param",
             "value": {"P": disturbance["P_disturbed"]}, "label": "CPL disturbance applied"},
            {"t": disturbance["t_clear"], "type": "param",
             "value": {"P": disturbance["P_nom"]}, "label": "CPL disturbance cleared"},
        ]

    sim = Simulator(model, method=method, rtol=rtol, atol=atol)
    t0 = time.time()
    result = sim.simulate(
        x0=x0, t_span=(0.0, t_end), controller=controller,
        disturbances=dist_events, n_eval=n_eval,
    )
    wall = time.time() - t0

    t = result.t
    X = result.x  # (4, N)
    i_l, v_b, sigma, v_o = X[0], X[1], X[2], X[3]

    # Manifold residual e_Sigma = v_o - v_nom + R_v i_l - K_i sigma.
    e_sigma = v_o - p["v_nom"] + p["R_v"] * i_l - p["K_i"] * sigma
    e0 = float(e_sigma[0])
    e_theory = e0 * np.exp(-p["k_m"] * t)
    resid_abs_err = np.abs(e_sigma - e_theory)
    max_resid_err = float(np.max(resid_abs_err)) if len(t) else float("nan")
    # Scale-independent measure of contraction-law agreement.
    denom = max(abs(e0), 1e-12)
    max_resid_rel_err = float(max_resid_err / denom)

    # CPL power vs time (for output/CSV), reconstructed from the schedule.
    cpl_power = np.full_like(t, P_nom)
    disturbance_active = np.zeros_like(t, dtype=int)
    if disturbance:
        mask = (t >= disturbance["t_start"]) & (t < disturbance["t_clear"])
        cpl_power[mask] = disturbance["P_disturbed"]
        disturbance_active[mask] = 1

    status = _classify_recovery(
        t, X, x_star, e_sigma, result, sim_cfg.get("recovery") or {}, p
    )

    run_id = uuid.uuid4().hex
    provenance = {
        "run_id": run_id,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "controller": "stabilizing_mrc (ideal, unsaturated, dv_o/dt = u)",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "solver": {
            "solver_name": method, "relative_tolerance": rtol,
            "absolute_tolerance": atol, "n_eval": n_eval,
            "time_span": [0.0, t_end], "dense_output_policy": "uniform t_eval grid",
            "wall_time_s": wall,
        },
        "parameters": p,
        "initial_condition": [float(v) for v in x0],
        "initial_perturbation": [float(v) for v in delta],
        "disturbance": disturbance,
        "equilibrium": {"x_star": [float(v) for v in x_star]},
    }

    dataset = {
        "run_id": run_id,
        "t": [float(v) for v in t],
        "i_l": [float(v) for v in i_l],
        "v_b": [float(v) for v in v_b],
        "sigma": [float(v) for v in sigma],
        "v_o": [float(v) for v in v_o],
        "e_sigma": [float(v) for v in e_sigma],
        "e_sigma_theory": [float(v) for v in e_theory],
        "cpl_power": [float(v) for v in cpl_power],
        "disturbance_active": [int(v) for v in disturbance_active],
        "x_star": {"i_l": float(x_star[0]), "v_b": float(x_star[1]),
                   "sigma": float(x_star[2]), "v_o": float(x_star[3])},
        "e_sigma_0": e0,
        "residual_contraction": {
            "max_abs_error": max_resid_err,
            "max_rel_error": max_resid_rel_err,
            "k_m": p["k_m"],
            "note": "Compares simulated e_Sigma(t) against e_Sigma(0)*exp(-k_m t) "
                    "(ideal unsaturated model identity).",
        },
        "events": [{"t": float(tt), "label": lbl} for (tt, lbl) in result.events],
        "status": status,
        "provenance": provenance,
    }
    return dataset


def _classify_recovery(t, X, x_star, e_sigma, result, rec_cfg, p) -> Dict:
    """
    Single-run recovery status vocabulary per reference section 8.2-8.3.
    Kept strictly distinct from local stability and from certified IMS.
    """
    dwell_frac = float(rec_cfg.get("dwell_frac", 0.2))
    eps_r = float(rec_cfg.get("eps_r", 1e-3))
    eps_x = float(rec_cfg.get("eps_x", 1e-2))
    v_min = float(rec_cfg.get("v_min", 1.0))
    M_div = float(rec_cfg.get("M_div", 1e6))

    solver_status = "completed" if result.success else "failed"
    termination_reason = result.message
    constraint_infeasible = False
    constraint_status = "within_domain"

    # Domain / voltage-floor violation (v_b <= v_min).
    v_b = X[1]
    floor_hit = np.any(v_b <= v_min)
    # Divergence monitor (physical-coordinate norm here; declared explicitly).
    dev = np.linalg.norm(X - x_star[:, None], axis=0)
    div_hit = np.any(dev > M_div)

    recovery_outcome = "Undefined"
    if not result.success:
        recovery_outcome = "Undefined"
        termination_reason = f"solver_failed: {result.message}"
    elif div_hit:
        recovery_outcome = "ThresholdTerminated"
        constraint_status = "monitoring_threshold_crossed"
        termination_reason = f"||x - x*|| exceeded M_div={M_div} (NOT proof of non-recovery)"
    elif floor_hit:
        recovery_outcome = "ConstraintViolated"
        constraint_status = f"bus voltage floor v_b <= v_min={v_min} crossed"
        termination_reason = "voltage-floor domain violation (not the mathematical singularity v_b=0)"
    else:
        # Evaluate dwell window [T-Td, T].
        T = t[-1]
        Td = dwell_frac * (T - t[0])
        win = t >= (T - Td)
        r_win = np.abs(e_sigma[win])
        x_win = X[:, win] - x_star[:, None]
        # Normalized by |x*| component-wise for eps_x comparison.
        scale = np.maximum(np.abs(x_star), 1.0)[:, None]
        xn = np.linalg.norm(x_win / scale, axis=0)
        manifold_recovered = bool(np.all(r_win <= eps_r))
        equilibrium_recovered = bool(manifold_recovered and np.all(xn <= eps_x))
        if equilibrium_recovered:
            recovery_outcome = "EquilibriumRecovered"
        elif manifold_recovered:
            recovery_outcome = "ManifoldRecovered"
        else:
            recovery_outcome = "NotRecoveredWithinHorizon"

    return {
        "solver_status": solver_status,
        "recovery_outcome": recovery_outcome,
        "termination_reason": termination_reason,
        "constraint_status": constraint_status,
        "constraint_infeasible": constraint_infeasible,
        "evaluated_until": float(t[-1]) if len(t) else 0.0,
        "criteria": {"dwell_frac": dwell_frac, "eps_r": eps_r, "eps_x": eps_x,
                     "v_min": v_min, "M_div": M_div, "metric": "physical (v_min, M_div); normalized-by-|x*| (eps_x)"},
        "disclaimer": (
            "Observed finite-horizon recovery is empirical evidence for THIS trajectory "
            "only. It is NOT a certified region of attraction and NOT a continuous-time "
            "proof (sampled on the solver's t_eval grid)."
        ),
    }


# ----------------------------------------------------------------------
# CSV export from the SAME dataset
# ----------------------------------------------------------------------
def dataset_to_csv(dataset: Dict) -> str:
    prov = dataset["provenance"]
    xs = dataset["x_star"]
    lines: List[str] = []
    lines.append(f"# model_id,{prov['model_id']}")
    lines.append(f"# model_version,{prov['model_version']}")
    lines.append(f"# run_id,{prov['run_id']}")
    lines.append(f"# timestamp_utc,{prov['timestamp_utc']}")
    lines.append(f"# solver,{prov['solver']['solver_name']},rtol,{prov['solver']['relative_tolerance']},atol,{prov['solver']['absolute_tolerance']}")
    lines.append(f"# x_star,i_l={xs['i_l']},v_b={xs['v_b']},sigma={xs['sigma']},v_o={xs['v_o']}")
    lines.append(f"# e_sigma_0,{dataset['e_sigma_0']}")
    if prov.get("disturbance"):
        d = prov["disturbance"]
        lines.append(f"# disturbance,P_nom={d['P_nom']},P_disturbed={d['P_disturbed']},t_start={d['t_start']},t_clear={d['t_clear']}")
    lines.append("time,i_l,v_b,sigma,v_o,e_sigma,e_sigma_theory,cpl_power,disturbance_active")
    n = len(dataset["t"])
    for k in range(n):
        lines.append(
            f"{dataset['t'][k]:.12g},{dataset['i_l'][k]:.12g},{dataset['v_b'][k]:.12g},"
            f"{dataset['sigma'][k]:.12g},{dataset['v_o'][k]:.12g},{dataset['e_sigma'][k]:.12g},"
            f"{dataset['e_sigma_theory'][k]:.12g},{dataset['cpl_power'][k]:.12g},{dataset['disturbance_active'][k]}"
        )
    return "\n".join(lines) + "\n"
