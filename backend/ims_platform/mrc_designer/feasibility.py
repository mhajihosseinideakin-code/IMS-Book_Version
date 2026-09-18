"""
ims_platform.mrc_designer.feasibility
--------------------------------------

Numeric/analytic MRC feasibility for  xdot = f(x) + G(x) u  with manifold
residual phi(x). Computes A(x) = Dphi(x) G(x) and reports the control
authority, exactly as required before any synthesis.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import sympy

from ..core.system import DynamicalSystem

#: Models whose validated SYMBOLIC dynamics + manifold make analytic MRC
#: establishment possible. Everything else gets numeric feasibility only.
SUPPORTED_ANALYTIC_MODELS = ("stabilizing_mrc_closed_loop", "stabilizing_mrc", "converter_cpl_paper")


def _has_symbolic(system: DynamicalSystem) -> bool:
    try:
        xs, us, ps = system.symbolic_symbols()
        system.symbolic_dynamics(xs, us, ps)
        system.symbolic_manifold_constraint(xs, us, ps)
        return True
    except NotImplementedError:
        return False
    except Exception:
        return False


def control_affine_split(system: DynamicalSystem, x: np.ndarray, p: Dict,
                         prefer_symbolic: bool = True) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Return (f(x), G(x), method). f is (n,), G is (n, m). 'method' is
    'analytic' (symbolic, authoritative) or 'finite_difference' (numeric,
    diagnostic only).
    """
    x = np.asarray(x, dtype=float)
    m = len(system.input_names)
    n = system.n_states
    if prefer_symbolic and _has_symbolic(system):
        xs, us, ps = system.symbolic_symbols()
        dyn = sympy.Matrix(system.symbolic_dynamics(xs, us, ps))
        subs_p = {ps[k]: p[k] for k in ps if k in p}
        subs_x = {xs[i]: float(x[i]) for i in range(n)}
        f_expr = dyn.subs({u: 0 for u in us})
        f = np.array([float(sympy.N(e.subs(subs_p).subs(subs_x))) for e in f_expr])
        G = np.zeros((n, m))
        for j, u in enumerate(us):
            col = dyn.diff(u)
            G[:, j] = [float(sympy.N(e.subs(subs_p).subs(subs_x).subs({uu: 0 for uu in us}))) for e in col]
        return f, G, "analytic"
    # numeric finite-difference control-affine split (diagnostic only)
    u0 = np.zeros(m)
    f = np.asarray(system.dynamics(0.0, x, u0, p), dtype=float)
    G = np.zeros((n, m))
    h = 1e-6
    for j in range(m):
        uj = np.zeros(m); uj[j] = h
        G[:, j] = (np.asarray(system.dynamics(0.0, x, uj, p), dtype=float) - f) / h
    return f, G, "finite_difference"


def manifold_gradient(system: DynamicalSystem, x: np.ndarray, p: Dict
                      ) -> Tuple[Optional[np.ndarray], Optional[float], str, str]:
    """
    Return (Dphi (k, n), phi(x), method, note). Analytic only when the model
    provides symbolic_manifold_constraint; otherwise (None, None,
    'numeric_manifold', reason) -- a numeric distance-to-M residual is not an
    analytic controlled-target manifold and does NOT support auto-synthesis.
    """
    x = np.asarray(x, dtype=float)
    n = system.n_states
    if _has_symbolic(system):
        xs, us, ps = system.symbolic_symbols()
        phi = system.symbolic_manifold_constraint(xs, us, ps)
        subs = {ps[k]: p[k] for k in ps if k in p}
        subs.update({xs[i]: float(x[i]) for i in range(n)})
        subs.update({u: 0 for u in us})
        grad = sympy.Matrix([phi]).jacobian(sympy.Matrix(xs))
        Dphi = np.array([[float(sympy.N(grad[0, i].subs(subs))) for i in range(n)]])
        phi_val = float(sympy.N(phi.subs(subs)))
        return Dphi, phi_val, "analytic", "Analytic manifold residual phi(x) from validated symbolic definition."
    return (None, None, "numeric_manifold",
            "This assembled model has no validated symbolic manifold phi(x); only a numeric "
            "distance-to-manifold residual is available. Analytic MRC establishment is not possible.")


def feasibility_report(system: DynamicalSystem, x: np.ndarray, p: Dict,
                       limits: Optional[Dict] = None) -> Dict:
    """
    Full MRC feasibility report at operating point x. Reports dims, A(x),
    rank, conditioning, relative degree, singular flag, actuator-limit
    compatibility, and whether MRC is analytically established.
    """
    x = np.asarray(x, dtype=float)
    f, G, g_method = control_affine_split(system, x, p)
    Dphi, phi_val, phi_method, phi_note = manifold_gradient(system, x, p)

    m = G.shape[1]
    report: Dict = {
        "operating_point": [float(v) for v in x],
        "dim_x": int(system.n_states),
        "dim_u": int(m),
        "input_names": list(system.input_names),
        "G_method": g_method,
        "manifold_method": phi_method,
        "manifold_note": phi_note,
    }

    if Dphi is None:
        report.update({
            "dim_phi": None, "A": None, "rank_A": None, "condition_number": None,
            "relative_degree_one": None, "singular_or_illconditioned": None,
            "mrc_established": False, "establishment_basis": "none",
            "reason": ("MRC not established for this model/configuration: no validated analytic "
                       "controlled-target manifold phi(x). " + phi_note),
        })
        return report

    A = Dphi @ G                      # (k, m)
    k = A.shape[0]
    rank = int(np.linalg.matrix_rank(A, tol=1e-9))
    # condition number of A (or A A^T for wide A)
    try:
        sv = np.linalg.svd(A, compute_uv=False)
        smin = float(sv.min()); smax = float(sv.max())
        cond = float(smax / smin) if smin > 0 else float("inf")
    except Exception:
        sv = np.array([]); cond = float("inf")
    rel_deg_one = bool(rank >= 1 and np.any(np.abs(A) > 1e-12))
    full_row_rank = bool(rank == k)
    illcond = bool((not np.all(np.isfinite(A))) or cond > 1e8 or not full_row_rank)

    # actuator-limit compatibility (informational)
    actuator = None
    if limits:
        actuator = {"limits": limits, "note": "Reported for constraint monitoring; synthesis is unsaturated."}

    established = bool(g_method == "analytic" and phi_method == "analytic" and full_row_rank and rel_deg_one and cond <= 1e8)
    report.update({
        "dim_phi": int(k),
        "phi_value": phi_val,
        "A": [[float(v) for v in row] for row in A],
        "singular_values": [float(s) for s in sv],
        "rank_A": rank,
        "full_row_rank": full_row_rank,
        "condition_number": cond,
        "relative_degree_one": rel_deg_one,
        "singular_or_illconditioned": illcond,
        "actuator": actuator,
        "mrc_established": established,
        "establishment_basis": "analytic" if established else ("numeric_only" if g_method != "analytic" else "insufficient_authority"),
        "reason": ("Analytic MRC conditions satisfied: A(x)=Dphi.G has full row rank and is well "
                   "conditioned; control input enters the residual dynamics (relative degree one)."
                   if established else
                   "MRC not established analytically at this configuration. "
                   + ("G(x) is finite-difference (diagnostic only); analytic establishment requires validated symbolic dynamics. "
                      if g_method != "analytic" else "")
                   + ("A(x) is rank-deficient / ill-conditioned: the selected manifold is not controllable "
                      "through the available inputs at this point. " if illcond else "")),
    })
    return report
