"""
ims_platform.mrc_designer.auto_manifold
-----------------------------------------

Automatic construction of a controller-shaped MRC target manifold phi(x)
for an assembled Network Builder model, and the full mathematical pipeline
around it -- this module is the answer to the standing gap recorded in
`memory/SPEC.md` ("even with f(x),G(x) available, there is NO validated
signed controlled-target manifold phi(x) for arbitrary topologies") for
the one structural class of topology the platform's component library
currently supports: a directly duty-modulated converter (BuckModel /
BoostModel / BuckBoostModel, network.converter_topologies), in any
network, feeding any of the platform's load types.

Mathematical basis
-------------------
Every model in this file's scope is control-affine in the duty ratio d,
because d enters the converter's own averaged inductor equation linearly
(network.converter_topologies):

    Buck:        L di_L/dt = d*v_in - v_bus - R_L*i_L
    Boost:       L di_L/dt = v_in - (1-d)*v_bus - R_L*i_L
    Buck-Boost:  L di_L/dt = d*v_in - (1-d)*v_bus - R_L*i_L

The duty ratio therefore has *relative degree one on i_L itself* -- it
enters the very first differentiation of i_L, not of some downstream
state. This is structurally different from the reduced-order voltage-
source reference model (models.converter_cpl_paper /
models.stabilizing_mrc), where the actuated quantity is a free outer-loop
voltage state v_o (dv_o/dt = u) and the manifold is built on a *different*
branch's own quasi-steady relation (book Ch. "Foundations of Intrinsic
Manifold Stability", eq. e_m). For these duty-cycle topologies there is no
such free voltage state to build on, so the book's own e_m-style
construction does not apply -- but relative degree one on i_L directly
means a controller-shaped, integral-augmented target manifold analogous
to the book's stabilizing-MRC construction (Ch. "Stabilizing Manifold-
Reshaping Control") DOES apply, anchored on i_L instead of v_o:

    sigma_dot = v_nom - v_bus                              (new integral state)
    phi(x) := i_L - i_bias - K_i * sigma                   (controller-shaped residual)

This is exactly the "explicit inner current-control loop" route the book
itself identifies (ch. mrc-closed-loop-verification, closing remark) as
the open path to a stabilizing manifold when the electrical-constraint
construction's reduced dynamics is unstable -- generalized here to
topologies where the duty ratio is the *only* available control input in
the first place, rather than retrofitted onto the reduced-order model.

phi is affine in (i_L, sigma), so Dphi is constant in those two
coordinates and zero everywhere else; A(x) = Dphi(x) G(x) reduces to the
single scalar d(di_L/dt)/dd -- v_in/L (buck), v_bus/L (boost), or
(v_in+v_bus)/L (buck-boost) -- independent of every other state in the
network, however large. This is what makes automatic construction
possible for an ARBITRARY surrounding network (any bus topology, any load
mix): feasibility of this particular phi depends only on the actuated
converter's own two parameters/state (v_in, v_bus), never on the rest of
the assembled system.

What this module does NOT claim
---------------------------------
This is one candidate manifold family (controller-shaped, current-loop),
not a claim that every Network Builder model admits SOME valid phi.  A
network with no directly duty-modulated converter (e.g. only a
`ConstantSetpointController`/`FilteredVoltageSource`-style component, or
only algebraic components) is honestly reported NOT ESTABLISHED by this
family, with a load-bearing but not exhaustive scan across the whole
project of possible manifold constructions -- see `inspect_network`'s
`reason` field for the exact failed condition in every case.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import sympy

from ..core.simulator import Simulator
from ..network.controller import (
    AutoCurrentLoopMRCController,
    ConstantDutyController,
    TOPOLOGY_CODE,
    current_loop_mrc_duty,
)
from ..network.converter import Converter
from ..network.converter_topologies import BuckModel, BoostModel, BuckBoostModel
from ..network.network import Network
from ..network.assembler import AutomaticModelBuilder, AssembledNetworkSystem
from .designer import MRCNotEstablished

TOPOLOGY_MODELS = {"buck": BuckModel, "boost": BoostModel, "buckboost": BuckBoostModel}
_MODEL_TO_TOPOLOGY = {v: k for k, v in TOPOLOGY_MODELS.items()}

DEFAULT_MRC_PARAMS = {"K_i": 20.0, "k_m": 500.0, "i_bias": 0.0}


# ---------------------------------------------------------------------------
# Step 1-2: locate the actuated converter, construct the candidate manifold
# ---------------------------------------------------------------------------

def find_duty_modulated_converter(net: Network, converter_id: Optional[str] = None) -> Converter:
    """
    Locate the (unique, unless `converter_id` disambiguates) Converter in
    `net` whose electrical model is a directly duty-modulated topology
    (Buck/Boost/Buck-Boost). Raises MRCNotEstablished with the exact
    reason if none exists, or if several exist and `converter_id` was not
    given to disambiguate -- never silently picks one.
    """
    candidates = [
        c for c in net.components
        if isinstance(c, Converter) and type(c.electrical_model) in _MODEL_TO_TOPOLOGY
    ]
    if converter_id is not None:
        matches = [c for c in candidates if c.id == converter_id]
        if not matches:
            raise MRCNotEstablished(
                f"MRC synthesis not established: '{converter_id}' is not a directly duty-modulated "
                f"converter (Buck/Boost/Buck-Boost) in this network. Candidates: "
                f"{[c.id for c in candidates] or 'none'}."
            )
        return matches[0]
    if not candidates:
        raise MRCNotEstablished(
            "MRC synthesis not established: no physically valid signed invariant target constraint "
            "could be derived -- this network contains no directly duty-modulated converter "
            "(Buck/Boost/Buck-Boost). This construction's control-affine relative-degree-one relation "
            "requires the duty ratio to enter an inductor branch's own averaged KVL equation directly; "
            "no such branch is present."
        )
    if len(candidates) > 1:
        raise MRCNotEstablished(
            f"MRC synthesis not established: {len(candidates)} directly duty-modulated converters are "
            f"present ({[c.id for c in candidates]}) -- 'converter_id' must be given explicitly to avoid "
            f"silently choosing one."
        )
    return candidates[0]


def symbolic_diL_dt(topology: str, iL, vbus, d, v_in, L, R_L):
    """
    SymPy expression for L*di_L/dt's dividend... i.e. di_L/dt itself,
    duplicated in exact algebraic correspondence with
    network.converter_topologies' own validated local_dynamics() (kept
    honest by test_auto_manifold_mrc.py::test_symbolic_matches_numeric_topology,
    which finite-differences the REAL topology class and compares).
    """
    if topology == "buck":
        return (d * v_in - vbus - R_L * iL) / L
    if topology == "boost":
        return (v_in - (1 - d) * vbus - R_L * iL) / L
    if topology == "buckboost":
        return (d * v_in - (1 - d) * vbus - R_L * iL) / L
    raise MRCNotEstablished(
        f"MRC synthesis not established: no automatic current-loop MRC construction is implemented "
        f"for topology '{topology}'. Supported: {sorted(TOPOLOGY_MODELS)}."
    )


def derive_symbolic_law(topology: str, mrc_params: Dict) -> Dict:
    """
    Steps 2-4 of the pipeline, symbolically: construct phi(x) = i_L -
    i_bias - K_i*sigma, differentiate it along the converter's own averaged
    dynamics (Dphi(x)[f(x)+G(x)u] = -k_m*phi(x)), check relative degree
    (d(phi_dot)/dd not identically zero), and solve for d in closed form.

    Returns a dict with the symbolic manifold, its derivative w.r.t. d
    (== A(x), a scalar here), the solved control law, and both as LaTeX.
    Raises MRCNotEstablished (with the exact failed condition) if the
    relative-degree check fails or SymPy cannot isolate d.
    """
    iL, vbus, d, sigma = sympy.symbols("i_L v_bus d sigma", real=True)
    v_in, L, R_L = sympy.symbols("v_in L R_L", positive=True)
    v_nom, K_i, k_m, i_bias = sympy.symbols("v_nom K_i k_m i_bias", real=True)

    f_iL = symbolic_diL_dt(topology, iL, vbus, d, v_in, L, R_L)
    phi = iL - i_bias - K_i * sigma
    sigma_dot = v_nom - vbus

    # Dphi(x)[f(x)+G(x)u]: only the i_L and sigma coordinates have a
    # nonzero partial derivative of phi (d(phi)/d(i_L)=1, d(phi)/d(sigma)=
    # -K_i, zero everywhere else in the assembled network's state, however
    # large that state is) -- this is exactly what makes phi's feasibility
    # independent of the rest of the network.
    phi_dot = 1 * f_iL + (-K_i) * sigma_dot

    A_expr = sympy.diff(phi_dot, d)  # A(x) = Dphi(x) G(x), scalar here
    A_simplified = sympy.simplify(A_expr)
    if A_simplified == 0:
        raise MRCNotEstablished(
            f"MRC synthesis not established: rank(Dφ G) = 0 identically for topology '{topology}' -- "
            f"the duty ratio does not enter φ_dot at all (relative degree greater than one); MRC "
            f"synthesis in closed form is not possible for this (manifold, input) pair."
        )

    solutions = sympy.solve(sympy.Eq(phi_dot, -k_m * phi), d)
    if not solutions:
        raise MRCNotEstablished(
            f"MRC synthesis not established: SymPy could not isolate 'd' from φ_dot = -k_m φ "
            f"for topology '{topology}'; the relation is not solvable in closed form."
        )
    control_expr = sympy.simplify(solutions[0])

    return {
        "topology": topology,
        "phi_symbolic": phi,
        "phi_dot_symbolic": phi_dot,
        "A_symbolic": A_simplified,
        "control_expr": control_expr,
        "control_latex": sympy.latex(sympy.Eq(d, control_expr)),
        "phi_latex": sympy.latex(sympy.Eq(sympy.Symbol("phi"), phi)),
        "symbols": {"iL": iL, "vbus": vbus, "d": d, "sigma": sigma, "v_in": v_in, "L": L,
                    "R_L": R_L, "v_nom": v_nom, "K_i": K_i, "k_m": k_m, "i_bias": i_bias},
    }


# ---------------------------------------------------------------------------
# Step 3: feasibility over a neighborhood, not just one point
# ---------------------------------------------------------------------------

def feasibility_over_neighborhood(law: Dict, params: Dict, i_L0: float, v_bus0: float,
                                   n_samples: int = 12, radius_frac: float = 0.15) -> Dict:
    """
    Evaluate A(x) = d(phi_dot)/dd numerically (from the symbolic
    expression, lambdified -- an independent evaluation path from the
    runtime `current_loop_mrc_duty` formula) at the center point and at
    `n_samples` perturbed (i_L, v_bus) points within +-radius_frac of the
    center, and check sign consistency / boundedness away from zero --
    the "over the relevant operating neighborhood, not merely at one
    arbitrary point" requirement.
    """
    syms = law["symbols"]
    A_fn = sympy.lambdify((syms["iL"], syms["vbus"], syms["v_in"], syms["L"], syms["R_L"]),
                          law["A_symbolic"], "numpy")
    v_in = float(params["v_in"]); L = float(params["L"]); R_L = float(params["R_L"])

    rng = np.random.default_rng(0)
    points = [(i_L0, v_bus0)]
    for _ in range(n_samples):
        di = i_L0 * radius_frac * rng.uniform(-1, 1)
        dv = v_bus0 * radius_frac * rng.uniform(-1, 1)
        points.append((i_L0 + di, max(v_bus0 + dv, 1e-6)))

    values = [float(A_fn(iL, vb, v_in, L, R_L)) for (iL, vb) in points]
    values_arr = np.asarray(values)
    signs = np.sign(values_arr)
    sign_consistent = bool(np.all(signs == signs[0]) or np.all(values_arr == 0))
    min_abs = float(np.min(np.abs(values_arr)))

    return {
        "n_points_sampled": len(points),
        "A_values": values,
        "A_center": values[0],
        "sign_consistent": sign_consistent,
        "min_abs_value": min_abs,
        "relative_degree_one_over_neighborhood": bool(sign_consistent and min_abs > 1e-9),
    }


# ---------------------------------------------------------------------------
# Step 1 + 5: control-affine verification and independent G(x) cross-check
# ---------------------------------------------------------------------------

def verify_control_affine_and_G(topology: str, converter: Converter, net: Network,
                                 x_probe_bus: float, i_L_probe: float, u_names_bus_index: int = 0) -> Dict:
    """
    Independent, finite-difference verification (step 1 and step 5's "F(x,u)
    ~= f(x)+G(x)u" and "finite-difference check G(x)" requirements) that the
    real, assembled electrical model is control-affine in d, and that its
    numeric d(di_L/dt)/dd matches the closed-form topology formula -- using
    the REAL `ConverterModel.local_dynamics`, not the symbolic duplicate.
    """
    em = converter.electrical_model
    p = em.params
    h = 1e-6
    d0 = 0.5
    f0 = em.local_dynamics(x_probe_bus, np.array([i_L_probe]), d0, p)[0]
    f1 = em.local_dynamics(x_probe_bus, np.array([i_L_probe]), d0 + h, p)[0]
    f_minus = em.local_dynamics(x_probe_bus, np.array([i_L_probe]), d0 - h, p)[0]
    G_fd = float((f1 - f_minus) / (2 * h))
    # affine check: three colinear samples in d should reproduce f0 exactly
    # via linear interpolation if and only if local_dynamics is affine in d
    f_mid_check = (f1 + f_minus) / 2.0
    affine_residual = float(abs(f_mid_check - f0))

    v_in = float(p["v_in"]); L = float(p["L"]); R_L = float(p["R_L"])
    if topology == "buck":
        G_closed_form = v_in / L
    elif topology == "boost":
        G_closed_form = x_probe_bus / L
    elif topology == "buckboost":
        G_closed_form = (v_in + x_probe_bus) / L
    else:
        G_closed_form = float("nan")

    return {
        "control_affine_verified": bool(affine_residual < 1e-6 * max(1.0, abs(f0))),
        "affine_residual": affine_residual,
        "G_finite_difference": G_fd,
        "G_closed_form": G_closed_form,
        "G_relative_error": float(abs(G_fd - G_closed_form) / max(abs(G_closed_form), 1e-12)),
    }


# ---------------------------------------------------------------------------
# Attaching the derived controller to the real Converter / Network
# ---------------------------------------------------------------------------

def _resolved_mrc_params(topology: str, converter: Converter, v_nom: float, mrc_params: Optional[Dict]) -> Dict:
    p = dict(DEFAULT_MRC_PARAMS)
    if mrc_params:
        p.update(mrc_params)
    em_p = converter.electrical_model.params
    return {
        "v_nom": float(v_nom),
        "K_i": float(p["K_i"]),
        "k_m": float(p["k_m"]),
        "i_bias": float(p["i_bias"]),
        "topology_code": TOPOLOGY_CODE[topology],
        "v_in": float(em_p["v_in"]),
        "L": float(em_p["L"]),
        "R_L": float(em_p.get("R_L", 0.0)),
    }


def attach_auto_mrc(net: Network, converter_id: str, v_nom: float, mrc_params: Optional[Dict] = None) -> Tuple[Network, Converter]:
    """
    Return a NEW Network (the original is left untouched) in which the
    target converter's controller has been replaced by
    `AutoCurrentLoopMRCController`, configured with the derived MRC
    parameters -- MRC becomes the converter's actual active controller,
    not merely an analysis-time overlay (task requirement 7).
    """
    converter = find_duty_modulated_converter(net, converter_id)
    topology = _MODEL_TO_TOPOLOGY[type(converter.electrical_model)]
    resolved = _resolved_mrc_params(topology, converter, v_nom, mrc_params)

    new_net = Network(net.name)
    for bus in net.buses.values():
        new_net.buses[bus.id] = bus
    new_net.branches = list(net.branches)
    new_net.components = []
    for comp in net.components:
        if comp.id == converter.id:
            new_controller = AutoCurrentLoopMRCController(**resolved)
            new_conv = Converter(id=comp.id, bus=comp.bus, electrical_model=comp.electrical_model,
                                  controller=new_controller)
            new_net.components.append(new_conv)
        else:
            new_net.components.append(comp)
    new_converter = next(c for c in new_net.components if c.id == converter.id)
    return new_net, new_converter


def _baseline_controller_for(converter: Converter) -> "Controller":
    """
    The converter's ORIGINAL controller, held at its own current
    operating duty, used for the open-loop/baseline arm of before/after --
    matches the platform's existing convention (network_mrc_closed_loop)
    of comparing MRC against the converter's un-MRC'd behavior on the
    identical assembled network, not against an unrelated model.
    """
    return converter.controller


# ---------------------------------------------------------------------------
# Full pipeline: inspect -> feasibility -> synthesize -> verify
# ---------------------------------------------------------------------------

def run_auto_mrc_pipeline(net: Network, converter_id: Optional[str] = None,
                          v_nom: Optional[float] = None, mrc_params: Optional[Dict] = None) -> Dict:
    """
    Network Builder -> assembled nonlinear dynamics -> control-affine split
    -> controlled-target manifold construction -> feasibility -> synthesis
    -> independent verification -> closed-loop / reduced-dynamics stability.

    Never raises for an unsupported model: every MRCNotEstablished is
    caught and reported as a structured "not established" result with the
    exact mathematical reason, per task requirement 8.
    """
    report: Dict = {"status": None, "reason": None}
    try:
        converter = find_duty_modulated_converter(net, converter_id)
    except MRCNotEstablished as e:
        report.update(status="MRC_SYNTHESIS_NOT_ESTABLISHED", reason=str(e))
        return report

    topology = _MODEL_TO_TOPOLOGY[type(converter.electrical_model)]
    report["converter_id"] = converter.id
    report["topology"] = topology

    # v_nom: the voltage the integral action regulates the converter's own
    # bus to. If not given explicitly, use the network's own bus v_init as
    # the operating target (the same convention the project's equilibrium
    # stage already anchors to).
    bus = net.buses[converter.bus]
    v_nom_val = float(v_nom) if v_nom is not None else float(bus.v_init or 1.0)
    resolved_params = _resolved_mrc_params(topology, converter, v_nom_val, mrc_params)

    # --- Step 2-4: symbolic manifold construction + synthesis ------------
    try:
        law = derive_symbolic_law(topology, resolved_params)
    except MRCNotEstablished as e:
        report.update(status="MRC_SYNTHESIS_NOT_ESTABLISHED", reason=str(e))
        return report

    report["candidate_phi"] = "phi(x) = i_L - i_bias - K_i*sigma,  sigma_dot = v_nom - v_bus"
    report["dim_phi"] = 1
    report["phi_latex"] = law["phi_latex"]
    report["control_law_latex"] = law["control_latex"]
    report["A_symbolic"] = str(law["A_symbolic"])

    # --- Step 1: baseline open-loop equilibrium (before touching the controller) ---
    baseline_system = AutomaticModelBuilder.build(net, name=net.name)
    try:
        base_eq = baseline_system.find_equilibrium(baseline_system.initial_guess(), with_eigs=False)
    except Exception as e:  # pragma: no cover - defensive
        report.update(status="MRC_SYNTHESIS_NOT_ESTABLISHED",
                      reason=f"MRC synthesis not established: baseline equilibrium solve failed ({e}).")
        return report

    n_states = baseline_system.n_states
    bus_state_name = net.buses[converter.bus].state_name  # e.g. "v_bus", not the bare bus id
    i_L_idx = None
    for name, idx in zip(baseline_system.state_names, range(n_states)):
        if name == f"{converter.id}_em_i_L":
            i_L_idx = idx
    v_bus_idx = None
    for name, idx in zip(baseline_system.state_names, range(n_states)):
        if name == bus_state_name:
            v_bus_idx = idx
    if i_L_idx is None or v_bus_idx is None:  # pragma: no cover - defensive
        report.update(status="MRC_SYNTHESIS_NOT_ESTABLISHED",
                      reason="MRC synthesis not established: could not locate the converter's i_L or its "
                             "bus voltage in the assembled state vector.")
        return report

    i_L0 = float(base_eq.x_star[i_L_idx])
    v_bus0 = float(base_eq.x_star[v_bus_idx])
    report["dim_x"] = int(n_states)
    report["dim_u"] = 1
    report["baseline_equilibrium"] = {baseline_system.state_names[i]: float(base_eq.x_star[i]) for i in range(n_states)}
    report["baseline_equilibrium_converged"] = bool(base_eq.converged)

    # --- Step 3: feasibility over a neighborhood --------------------------
    feas = feasibility_over_neighborhood(law, resolved_params, i_L0, v_bus0)
    report["feasibility"] = feas
    report["rank_A"] = 1 if feas["relative_degree_one_over_neighborhood"] else 0
    report["relative_degree"] = 1 if feas["relative_degree_one_over_neighborhood"] else None

    # --- Step 1 + 5: independent finite-difference cross-check of G(x) ---
    ga = verify_control_affine_and_G(topology, converter, net, v_bus0, i_L0)
    report["control_affine_verification"] = ga

    if not feas["relative_degree_one_over_neighborhood"]:
        report.update(
            status="MRC_SYNTHESIS_NOT_ESTABLISHED",
            reason=(f"MRC synthesis not established: rank(Dφ G) = 0 (or sign-indefinite) somewhere in "
                    f"the sampled operating neighborhood around i_L*={i_L0:.6g}, v_bus*={v_bus0:.6g} "
                    f"(min|A(x)|={feas['min_abs_value']:.3e}); relative degree one does not hold uniformly."),
        )
        return report
    if not ga["control_affine_verified"] or ga["G_relative_error"] > 1e-4:
        report.update(
            status="MRC_SYNTHESIS_NOT_ESTABLISHED",
            reason=(f"MRC synthesis not established: independent finite-difference verification of G(x) "
                    f"disagrees with the closed-form derivation (relative error "
                    f"{ga['G_relative_error']:.3e}); the assumed control-affine structure is not confirmed "
                    f"on the real assembled model."),
        )
        return report

    # --- Attach + closed-loop equilibrium/eigenstructure ------------------
    mrc_net, mrc_converter = attach_auto_mrc(net, converter.id, v_nom_val, mrc_params)
    mrc_system = AutomaticModelBuilder.build(mrc_net, name=net.name + "_auto_mrc")
    x0_guess = mrc_system.initial_guess()
    # seed the guess from the baseline equilibrium where state names match
    name_to_idx_mrc = {n: i for i, n in enumerate(mrc_system.state_names)}
    for name, idx in zip(baseline_system.state_names, range(n_states)):
        if name in name_to_idx_mrc:
            x0_guess[name_to_idx_mrc[name]] = base_eq.x_star[idx]
    cl_eq = mrc_system.find_equilibrium(x0_guess, with_eigs=True)
    report["closed_loop_equilibrium"] = {mrc_system.state_names[i]: float(cl_eq.x_star[i]) for i in range(mrc_system.n_states)}
    report["closed_loop_equilibrium_converged"] = bool(cl_eq.converged)
    report["closed_loop_equilibrium_residual_norm"] = float(cl_eq.residual_norm)

    # Step 3's "equilibrium compatibility" check: a Newton failure here is
    # not a numerical accident -- it is the model saying the requested
    # v_nom is not an equilibrium this topology/parameter set can reach
    # (e.g. a target below v_in for a Boost, or above v_in for a Buck,
    # outside the ideal-CCM-average achievable range for any duty in
    # [0.02, 0.98]). Report NOT ESTABLISHED with that exact reason rather
    # than continuing to report eigenvalues/contraction numbers computed
    # at a non-equilibrium point.
    if not cl_eq.converged:
        report.update(
            status="MRC_SYNTHESIS_NOT_ESTABLISHED",
            reason=(f"MRC synthesis not established: target manifold incompatible with equilibrium -- the "
                    f"closed-loop Newton solve did not converge (residual norm {cl_eq.residual_norm:.3e}) "
                    f"for v_nom={v_nom_val:.6g} on topology '{topology}' with v_in="
                    f"{resolved_params['v_in']:.6g}. This target bus voltage is likely not reachable by "
                    f"this topology at any admissible duty d in [0.02, 0.98] (e.g. a Boost cannot regulate "
                    f"below v_in, nor a Buck above it, in the ideal averaged model)."),
        )
        return report

    eigs = cl_eq.eigenvalues
    k_m = resolved_params["k_m"]
    idx_transverse = int(np.argmin(np.abs(eigs - (-k_m)))) if eigs is not None else None
    transverse = eigs[idx_transverse] if idx_transverse is not None else None
    tangential = [complex(z) for i, z in enumerate(eigs) if i != idx_transverse] if eigs is not None else []
    report["closed_loop_eigenvalues"] = [{"re": float(z.real), "im": float(z.imag)} for z in eigs] if eigs is not None else None
    report["transverse_eigenvalue"] = {"re": float(transverse.real), "im": float(transverse.imag)} if transverse is not None else None
    report["transverse_matches_km"] = bool(transverse is not None and abs(transverse.real + k_m) < max(1e-2 * k_m, 1.0) and abs(transverse.imag) < 1.0)
    report["tangential_eigenvalues"] = [{"re": float(z.real), "im": float(z.imag)} for z in tangential]
    report["tangential_locally_stable"] = bool(all(z.real < 0 for z in tangential)) if tangential else None

    # --- Step 5-6: contraction residual + separate transverse/tangential claims ---
    contraction = _contraction_residual_check(mrc_system, mrc_converter, resolved_params, cl_eq.x_star,
                                              name_to_idx_mrc, bus_state_name)
    report["contraction_residual"] = contraction

    slowest_real = min((abs(z.real) for z in tangential if abs(z.real) > 1e-9), default=k_m)
    # >=8 time constants of the slowest tangential mode, but capped: a
    # near-marginal tangential mode (|Re(lambda)| -> 0, e.g. an
    # undamped LC reduced dynamic under a constant-current load) would
    # otherwise blow this horizon up to an astronomically large value
    # and make the diagnostic recovery simulation below run for an
    # effectively unbounded wall-clock time. Capping it means the
    # "observed_finite_horizon_recovery" claim (book: a bounded,
    # finite-horizon observation, never promoted to asymptotic/regional
    # IMS on its own) is reported honestly as "not observed within this
    # capped horizon" rather than the pipeline hanging.
    horizon_uncapped = max(0.05, 8.0 / slowest_real)
    horizon_cap = 2.0
    horizon = min(horizon_uncapped, horizon_cap)
    bus_idx_mrc = name_to_idx_mrc[bus_state_name]
    sim_result = _closed_loop_simulation(mrc_system, cl_eq.x_star, t_end=horizon, perturb_idx=bus_idx_mrc)
    sim_result["horizon_capped"] = bool(horizon_uncapped > horizon_cap)
    sim_result["horizon_uncapped_time_constants_requested"] = float(horizon_uncapped)
    report["simulation_result"] = sim_result

    # --- Before/After (task step 7): identical disturbance, original
    # controller vs MRC, on the identical assembled network -----------------
    iL_idx_mrc = name_to_idx_mrc[f"{converter.id}_em_i_L"]
    report["closed_loop_comparison"] = _before_after_trajectories(
        baseline_system, base_eq.x_star, i_L_idx, v_bus_idx,
        mrc_system, cl_eq.x_star, iL_idx_mrc, bus_idx_mrc, horizon,
    )
    report["closed_loop_comparison"]["v_nom"] = v_nom_val

    report["derived_control_law"] = law["control_latex"]
    report["equilibrium_preservation"] = {
        "residual_norm": report["closed_loop_equilibrium_residual_norm"],
        "converged": report["closed_loop_equilibrium_converged"],
    }
    report["saturation_note"] = _saturation_check(mrc_converter, cl_eq.x_star, name_to_idx_mrc, converter.id,
                                                   bus_state_name)

    claim = {
        "ideal_residual_contraction": {
            "claim": "phi_dot = -k_m*phi within the ideal (unsaturated) model",
            "max_relative_error": contraction["max_relative_error"],
        },
        "local_equilibrium_stability": {
            "claim": "all closed-loop eigenvalues have negative real part (linearisation)",
            "established": report["tangential_locally_stable"] and report["transverse_matches_km"],
        },
        "observed_finite_horizon_recovery": {
            "claim": "simulated trajectory returns toward the equilibrium within the run horizon",
            "observed": sim_result.get("recovered"),
        },
        "certified_regional_ims": {
            "claim": "a stated region on which every reduced trajectory converges, with a uniform bound",
            "established": False,
            "note": "Not established by this pipeline (matches the book's own stabilizing-MRC scope: a "
                     "local Hurwitz-linearisation result is not a regional IMS certificate).",
        },
    }
    report["claim_levels"] = claim

    if report["tangential_locally_stable"] and report["transverse_matches_km"] and ga["control_affine_verified"]:
        report["status"] = "MRC_SYNTHESIS_SUPPORTED"
        report["reason"] = ("Analytic control-affine split verified independently, relative degree one "
                             "confirmed over the sampled neighborhood, closed-form MRC law derived and "
                             "solved, and the closed-loop equilibrium is locally exponentially stable "
                             "(all eigenvalues strictly negative real part).")
    else:
        report["status"] = "MRC_FEASIBILITY_DIAGNOSTIC_ONLY"
        report["reason"] = ("MRC was derived and the exact transverse residual identity holds within the "
                             "ideal model, but the reduced/tangential closed-loop dynamics is not confirmed "
                             "locally exponentially stable at this operating point -- reduced dynamics "
                             "stability (condition (iii)) is a separate claim from transverse contraction, "
                             "never inferred from it.")
    return report


def _contraction_residual_check(mrc_system: AssembledNetworkSystem, mrc_converter: Converter,
                                params: Dict, x_star: np.ndarray, name_to_idx: Dict,
                                bus_state_name: str) -> Dict:
    """
    Step 5: substitute u_MRC back into the ORIGINAL assembled equations and
    independently verify phi_dot + k_m*phi ~= 0, at the equilibrium, at a
    random nearby state, and at a disturbed state -- never inferred from
    "the trajectory converged".
    """
    conv_id = mrc_converter.id
    i_L_key = f"{conv_id}_em_i_L"
    sigma_key = f"{conv_id}_ctrl_sigma"
    iL_idx = name_to_idx[i_L_key]
    sigma_idx = name_to_idx[sigma_key]
    bus_idx = name_to_idx[bus_state_name]
    i_bias = params["i_bias"]; K_i = params["K_i"]; k_m = params["k_m"]

    def phi_of(x: np.ndarray) -> float:
        return float(x[iL_idx] - i_bias - K_i * x[sigma_idx])

    def phi_dot_numeric(x: np.ndarray) -> float:
        f = np.asarray(mrc_system.dynamics(0.0, x, mrc_system.default_input(), mrc_system.params), dtype=float)
        return float(f[iL_idx] - K_i * f[sigma_idx])

    def eval_point(x: np.ndarray, label: str) -> Dict:
        phi = phi_of(x)
        phi_dot = phi_dot_numeric(x)
        residual = phi_dot + k_m * phi
        return {"label": label, "phi": phi, "phi_dot": phi_dot, "residual": residual,
                "residual_abs": abs(residual)}

    rng = np.random.default_rng(1)
    x_nearby = x_star.copy()
    x_nearby[iL_idx] += 0.03 * max(abs(x_star[iL_idx]), 1.0) * rng.uniform(-1, 1)
    x_nearby[bus_idx] += 0.03 * max(abs(x_star[bus_idx]), 1.0) * rng.uniform(-1, 1)
    x_disturbed = x_star.copy()
    x_disturbed[bus_idx] *= 0.85  # a 15% bus-voltage sag, off the manifold

    points = [eval_point(x_star, "equilibrium"), eval_point(x_nearby, "random_nearby"),
              eval_point(x_disturbed, "disturbed")]
    max_rel = max(p["residual_abs"] / max(abs(k_m * p["phi"]), 1e-9) if abs(p["phi"]) > 1e-9 else p["residual_abs"]
                  for p in points)
    return {"points": points, "max_relative_error": float(max_rel),
            "identity": "phi_dot + k_m*phi ~= 0 (ideal, unsaturated model)"}


def _closed_loop_simulation(mrc_system: AssembledNetworkSystem, x_star: np.ndarray, t_end: float = 0.05,
                             perturb_idx: int = 0, return_trajectory: bool = False) -> Dict:
    x0 = x_star.copy()
    x0[perturb_idx] *= 0.85  # perturb the target converter's own bus voltage (a 15% sag)
    init_dev = float(np.linalg.norm(x0 - x_star))
    sim = Simulator(mrc_system, method="RK45")
    try:
        # max_step bounds RK45 to a sane fraction of the horizon so a
        # near-singular / stiff perturbed trajectory (duty law
        # denominator approached but not crossed, marginal tangential
        # dynamics, etc.) fails fast with solve_ivp's own step-size-
        # too-small diagnostic instead of the adaptive stepper grinding
        # through an effectively unbounded number of tiny steps.
        result = sim.simulate(x0=x0, t_span=(0.0, t_end), n_eval=500, max_step=max(t_end / 200.0, 1e-6))
    except (ValueError, FloatingPointError, ZeroDivisionError) as e:
        # A genuine physical/numerical breakdown along the perturbed
        # trajectory (e.g. the duty law's own denominator crossing zero
        # because the state left the domain the linearisation is valid
        # near) -- reported as a solver failure, exactly the book's own
        # solver_status = SolverFailure category
        # (ch:recoverability-theory), never silently swallowed or
        # allowed to crash the whole pipeline.
        out = {"t_end": t_end, "success": False, "message": f"solver failure: {e}",
               "initial_deviation": init_dev, "final_deviation": None, "recovered": False}
        if return_trajectory:
            out["t"] = None; out["x"] = None
        return out
    final_dev = float(np.linalg.norm(result.x[:, -1] - x_star))
    out = {"t_end": t_end, "success": bool(result.success), "message": result.message,
           "initial_deviation": init_dev, "final_deviation": final_dev,
           "recovered": bool(result.success and final_dev < 0.1 * init_dev)}
    if return_trajectory:
        out["t"] = result.t.tolist()
        out["x"] = result.x.tolist()
    return out


def _before_after_trajectories(baseline_system: AssembledNetworkSystem, base_eq_x_star: np.ndarray,
                                base_iL_idx: int, base_bus_idx: int,
                                mrc_system: AssembledNetworkSystem, mrc_eq_x_star: np.ndarray,
                                mrc_iL_idx: int, mrc_bus_idx: int, t_end: float) -> Dict:
    """
    Task step 7's "Before/After" report, made concrete: the SAME 15% bus-
    voltage sag applied to (a) the original assembled network with its
    original (pre-MRC) controller, and (b) the identical network with the
    automatically-derived MRC now the converter's active controller --
    full i_L(t)/v_bus(t)/deviation-from-equilibrium trajectories for both,
    for a genuine baseline-vs-MRC comparison rather than a single-run
    diagnostic.
    """
    base_result = _closed_loop_simulation(baseline_system, base_eq_x_star, t_end=t_end,
                                           perturb_idx=base_bus_idx, return_trajectory=True)
    mrc_result = _closed_loop_simulation(mrc_system, mrc_eq_x_star, t_end=t_end,
                                          perturb_idx=mrc_bus_idx, return_trajectory=True)

    def extract(res: Dict, iL_idx: int, bus_idx: int) -> Dict:
        out = dict(res)
        if res.get("x") is not None:
            x_arr = np.array(res["x"])
            out["i_L"] = x_arr[iL_idx, :].tolist()
            out["v_bus"] = x_arr[bus_idx, :].tolist()
        else:
            out["i_L"] = None
            out["v_bus"] = None
        out.pop("x", None)
        return out

    return {"baseline": extract(base_result, base_iL_idx, base_bus_idx),
            "mrc": extract(mrc_result, mrc_iL_idx, mrc_bus_idx)}


def _saturation_check(mrc_converter: Converter, x_star: np.ndarray, name_to_idx: Dict,
                      conv_id: str, bus_state_name: str) -> Dict:
    iL = float(x_star[name_to_idx[f"{conv_id}_em_i_L"]])
    sigma = float(x_star[name_to_idx[f"{conv_id}_ctrl_sigma"]])
    vbus = float(x_star[name_to_idx[bus_state_name]])
    p = mrc_converter.controller.params
    d_raw = current_loop_mrc_duty("", iL, vbus, sigma, p)
    saturated = bool(d_raw < 0.02 or d_raw > 0.98)
    return {"duty_at_equilibrium": d_raw, "saturated": saturated,
            "note": ("Duty at equilibrium is within [0.02, 0.98]: the exact contraction identity is "
                     "active there." if not saturated else
                     "Duty at equilibrium falls outside the converter's clamp [0.02, 0.98]: the exact "
                     "contraction identity phi_dot=-k_m*phi is NOT active while the duty clamp is engaged.")}
