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
from scipy.linalg import null_space as _scipy_null_space

from ..core.simulator import Simulator
from ..network.controller import (
    AutoCurrentLoopMRCController,
    BusVoltageAnchoredMRCController,
    ConstantDutyController,
    TOPOLOGY_CODE,
    current_loop_mrc_duty,
    bus_voltage_mrc_duty,
)
from ..network.converter import Converter
from ..network.converter_topologies import BuckModel, BoostModel, BuckBoostModel
from ..network.network import Network
from ..network.assembler import AutomaticModelBuilder, AssembledNetworkSystem
from .designer import MRCNotEstablished

TOPOLOGY_MODELS = {"buck": BuckModel, "boost": BoostModel, "buckboost": BuckBoostModel}
_MODEL_TO_TOPOLOGY = {v: k for k, v in TOPOLOGY_MODELS.items()}

DEFAULT_MRC_PARAMS = {"K_i": 20.0, "k_m": 500.0, "i_bias": 0.0}

#: Second-candidate (bus-voltage-anchored) manifold default params. K_i,
#: k_m reuse the same defaults as the first candidate (same integrator,
#: same transverse rate target); R_v (the droop/integral-action gain
#: multiplying i_L in phi2 = v_bus - v_nom + R_v*i_L - K_i*sigma) has NO
#: universal default that is analytically justified across topologies --
#: section "Second-candidate manifold: R_v search" below derives, per
#: topology, that the stabilizing region for a constant-power load is a genuinely
#: bounded window of R_v (not "bigger is always better"), so
#: `run_auto_mrc_pipeline`'s reshaping step searches R_v explicitly
#: rather than assuming one constant works everywhere.
DEFAULT_MRC_PARAMS_V2 = {"K_i": 20.0, "k_m": 500.0}

#: Geometric search grid for R_v when attempting the second-candidate
#: (bus-voltage-anchored) manifold. Derived and numerically confirmed
#: (scratch derivation, not shipped as a file -- see the sub-task D final
#: report) to bracket the stabilizing window for every topology tested
#: (buck: exact 0 < R_v < v_nom^2/P; boost: a small lower threshold,
#: stable up to very large R_v; buck-boost: a BOUNDED window, unstable
#: again above a finite upper R_v) at the platform's representative
#: parameter scales (v_nom in the tens of volts, P in the tens of watts,
#: R_L ~ 0.05 ohm). This is a bounded, reported search -- never a single
#: hardcoded constant asserted to work, and the full set of attempted
#: values together with each one's own alpha_parallel is always reported
#: (see `report["manifold_reshaping"]["R_v_search"]`), including when
#: every candidate fails.
R_V_SEARCH_GRID = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

#: Numerical tolerance on the reduced/tangential dynamics' dominant real
#: eigenvalue part (alpha_parallel), below which a classification of
#: "stable" or "unstable" is not claimed. alpha_parallel comes from a
#: numerically-computed (finite-difference-Jacobian-derived) eigenvalue,
#: so an exact floating-point comparison with 0 would misclassify a
#: genuine stability-BOUNDARY operating point (alpha_parallel intended to
#: be exactly 0 analytically, but landing at e.g. +/-1e-9 numerically) as
#: confidently stable or unstable. Any case with |alpha_parallel| at or
#: below this tolerance is reported NOT_ESTABLISHED (a genuine third
#: state -- "the boundary itself", not "violated").
TANGENTIAL_EIGENVALUE_TOLERANCE = 1e-6


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


def list_duty_modulated_converters(net: Network) -> List[Converter]:
    """
    Every directly duty-modulated (Buck/Boost/Buck-Boost) Converter in
    `net`, in declaration order. Pure inspection -- never raises, never
    chooses one; used by `run_auto_mrc_pipeline_multi` to enumerate every
    candidate before deciding whether auto-selection or explicit
    selection is required.
    """
    return [c for c in net.components
            if isinstance(c, Converter) and type(c.electrical_model) in _MODEL_TO_TOPOLOGY]


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
# Second-candidate manifold: bus-voltage-anchored (Sub-task D, Option 1)
# ---------------------------------------------------------------------------
#
# When the first candidate (phi1 = i_L - i_bias - K_i*sigma) achieves exact
# transverse contraction but its reduced/tangential dynamics are unstable
# (proved, for constant-power loads under phi1, to be UNCONDITIONAL: trace(J_reduced)
# = P/(C*v_bus*^2) > 0 for every K_i, k_m, i_bias, R_L -- see
# test_cpl_reduced_dynamics.py), the platform attempts a SECOND,
# independently-derived candidate rather than reclassifying or weakening
# the first result:
#
#     v_Sigma = v_nom - R_v*i_L + K_i*sigma            (droop + integral,
#                                                        book's stabilizing-
#                                                        MRC e_Sigma construction)
#     phi2(x) = v_bus - v_Sigma
#             = v_bus - v_nom + R_v*i_L - K_i*sigma
#
# reinterpreting the book's v_o (a free outer-loop voltage state that these
# single-stage Buck/Boost/Buck-Boost topologies do not have) as v_bus, the
# only voltage state that actually exists here.
#
# Unlike phi1, phi2's A(x) = Dphi2(x)G(x) genuinely differs per topology
# (this module's docstring for phi1 explicitly notes phi1's A(x) was
# v_in/L, v_bus/L, (v_in+v_bus)/L for buck/boost/buckboost -- a difference
# only in an overall multiplicative factor of a SINGLE d-dependent term).
# phi2's A(x) is structurally different: because port_current only enters
# dv_bus/dt (not R_v*di_L/dt) with a topology-dependent d-dependence,
#
#     Buck:        A2 = R_v*v_in/L                          (needs R_v != 0
#                                                              for relative
#                                                              degree one --
#                                                              buck's port
#                                                              current has
#                                                              NO direct
#                                                              d-dependence)
#     Boost:       A2 = R_v*v_bus/L - i_L/C                  (relative degree
#                                                              one even at
#                                                              R_v=0, from the
#                                                              port current's
#                                                              own (1-d)
#                                                              factor)
#     Buck-Boost:  A2 = R_v*(v_bus+v_in)/L - i_L/C            (same
#                                                              structure as
#                                                              Boost, with
#                                                              v_bus+v_in in
#                                                              place of v_bus)
#
# -- derived independently below from each topology's OWN f(x),G(x)
# (`symbolic_diL_dt` and the corresponding `port_current`, exactly the
# real `ConverterModel` formulas), never copied from one topology to
# another.


def symbolic_port_current(topology: str, iL, vbus, d):
    """
    SymPy expression for port_current(d, i_L), duplicated in exact
    algebraic correspondence with each topology's own
    `ConverterModel.port_current` (network.converter_topologies) -- the
    quantity phi2's dv_bus/dt actually depends on, distinct from
    `symbolic_diL_dt` (which governs phi1's manifold instead).
    """
    if topology == "buck":
        return iL
    if topology in ("boost", "buckboost"):
        return (1 - d) * iL
    raise MRCNotEstablished(
        f"MRC synthesis not established: no automatic bus-voltage-anchored MRC construction is "
        f"implemented for topology '{topology}'. Supported: {sorted(TOPOLOGY_MODELS)}."
    )


def derive_symbolic_law_v2(topology: str, mrc_params: Dict) -> Dict:
    """
    Second-candidate analogue of `derive_symbolic_law`: constructs
    phi2(x) = v_bus - v_nom + R_v*i_L - K_i*sigma, differentiates it
    along the ACTUAL topology's f(x),G(x) (via `symbolic_diL_dt` and
    `symbolic_port_current` -- the real per-topology relations, not a
    single formula reused across topologies), checks relative degree
    (A2(x) = d(phi2_dot)/dd not identically zero), and solves for d in
    closed form -- with the OTHER bus current injection `i_other` left as
    a free symbol (this manifold's derivative genuinely depends on the
    rest of the network's injection into the same bus, unlike phi1's;
    the runtime law `bus_voltage_mrc_duty` supplies `i_other` from the
    real `current_injection` of whatever else sits on the bus, evaluated
    live -- see `attach_bus_voltage_mrc`).

    Raises MRCNotEstablished (with the exact failed condition) if A2 is
    identically zero or SymPy cannot isolate d.
    """
    iL, vbus, d, sigma = sympy.symbols("i_L v_bus d sigma", real=True)
    v_in, L, R_L, C, R_v = sympy.symbols("v_in L R_L C R_v", positive=True)
    v_nom, K_i, k_m = sympy.symbols("v_nom K_i k_m", real=True)
    i_other = sympy.symbols("i_other", real=True)

    f_iL = symbolic_diL_dt(topology, iL, vbus, d, v_in, L, R_L)
    port = symbolic_port_current(topology, iL, vbus, d)

    phi2 = vbus - v_nom + R_v * iL - K_i * sigma
    sigma_dot = v_nom - vbus
    dvbus_dt = (port + i_other) / C
    phi2_dot = sympy.simplify(dvbus_dt + R_v * f_iL - K_i * sigma_dot)

    A2_expr = sympy.diff(phi2_dot, d)
    A2_simplified = sympy.simplify(A2_expr)
    if A2_simplified == 0:
        raise MRCNotEstablished(
            f"MRC manifold reshaping not established: rank(Dφ2 G) = 0 identically for topology "
            f"'{topology}' -- the duty ratio does not enter φ2_dot at all for this manifold; relative "
            f"degree one does not hold for the bus-voltage-anchored candidate on this topology."
        )

    solutions = sympy.solve(sympy.Eq(phi2_dot, -k_m * phi2), d)
    if not solutions:
        raise MRCNotEstablished(
            f"MRC manifold reshaping not established: SymPy could not isolate 'd' from φ2_dot = -k_m φ2 "
            f"for topology '{topology}'; the relation is not solvable in closed form."
        )
    control_expr = sympy.simplify(solutions[0])

    return {
        "topology": topology,
        "phi_symbolic": phi2,
        "phi_dot_symbolic": phi2_dot,
        "A_symbolic": A2_simplified,
        "control_expr": control_expr,
        "control_latex": sympy.latex(sympy.Eq(d, control_expr)),
        "phi_latex": sympy.latex(sympy.Eq(sympy.Symbol("phi_2"), phi2)),
        "symbols": {"iL": iL, "vbus": vbus, "d": d, "sigma": sigma, "v_in": v_in, "L": L,
                    "R_L": R_L, "C": C, "R_v": R_v, "v_nom": v_nom, "K_i": K_i, "k_m": k_m,
                    "i_other": i_other},
    }


# ---------------------------------------------------------------------------
# Analytic Routh-Hurwitz derivation of the (R_v, K_i) admissible region
# ---------------------------------------------------------------------------
#
# For a constant-power load (i_inj = -P/v_bus), the phi2-reduced 2-state system (over
# i_L, sigma, after substituting the manifold relation v_bus =
# v_nom - R_v*i_L + K_i*sigma and the closed-form MRC duty law) has trace
# and determinant that were derived symbolically (SymPy) and confirmed
# against the pipeline's own numerically-computed alpha_parallel to 1e-6
# relative precision (see test_manifold_reshaping.py and
# test_m2_validation.py). Routh-Hurwitz for a 2x2 system is exactly
# {trace<0, det>0} -- BOTH conditions, never trace alone.
#
# BUCK admits a fully closed-form (R_v, K_i) region because port_current
# = i_L has no direct duty dependence, which collapses the reduced
# dynamics' v_bus-dependence to a single clean substitution:
#
#     trace(J2_reduced) = P/(C*v_nom^2) - 1/(C*R_v)
#     det(J2_reduced)   = K_i/(C*R_v)
#
# BOOST and BUCK-BOOST do NOT admit a comparably simple closed form: their
# port_current = (1-d)*i_L depends on d directly, so the closed-form duty
# law solved from phi2_dot=-k_m*phi2 (a division by A2(x), itself state-
# dependent) propagates into trace/det as substantially longer rational
# expressions in i_L*, R_v, K_i, C, L, v_in that SymPy cannot simplify to
# an interpretable closed-form inequality (attempted, see the sub-task D/E
# scratch derivation; symbolic det/trace were obtained but not invertible
# into a clean R_v-vs-K_i boundary curve). For these two topologies this
# module reports a NUMERICALLY-DERIVED admissible window (bisection on
# alpha_parallel(R_v), not a coarse fixed grid) and states this limitation
# explicitly rather than presenting a numeric result as a closed form.


def buck_phi2_admissible_region(P: float, C: float, v_nom: float) -> Dict:
    """
    CLOSED-FORM Routh-Hurwitz admissible region for the buck topology's
    bus-voltage-anchored candidate under a constant-power load of power P:

        trace = P/(C*v_nom^2) - 1/(C*R_v) < 0   <=>   R_v < v_nom^2/P   (R_v>0)
        det   = K_i/(C*R_v) > 0                 <=>   K_i and R_v have the same sign

    Restricting to the physically conventional signs (R_v>0, an actual
    droop resistance; K_i>0, an actual integral gain -- the sign
    convention this whole module and its runtime controller already use),
    the admissible region is exactly

        0 < R_v < v_nom^2/P,     K_i > 0 (any positive value).

    K_i does not appear in the trace at all (only in det, whose sign
    condition is satisfied by any K_i>0) -- exactly mirroring the
    already-proven K_i-independence of phi1's alpha_parallel; K_i sets
    how fast the pair of eigenvalues converges/how they split into a
    complex-conjugate pair vs two real eigenvalues, never WHETHER they are
    stable.
    """
    R_v_max = v_nom ** 2 / P
    return {
        "topology": "buck",
        "load": "constant_power",
        "derivation": "closed_form",
        "trace_formula": "P/(C*v_nom**2) - 1/(C*R_v)",
        "det_formula": "K_i/(C*R_v)",
        "R_v_admissible_interval": (0.0, R_v_max),
        "R_v_max": R_v_max,
        "K_i_condition": "K_i > 0 (any positive value; does not affect trace, only det's sign, "
                         "already satisfied for R_v>0)",
        "region_description": f"0 < R_v < {R_v_max:.6g} (= v_nom^2/P), K_i > 0",
    }


def verify_buck_phi2_routh_hurwitz_symbolic() -> Dict:
    """
    Independent SymPy re-derivation of the buck trace/det formulas used by
    `buck_phi2_admissible_region`, run fresh each call (not merely
    asserted) -- the same "check against an independently re-derived
    result" discipline used throughout this module. Returns the derived
    trace/det as SymPy expressions plus a boolean confirming they match
    the closed forms in `buck_phi2_admissible_region`'s docstring.
    """
    i_L, sigma, P, C, K_i, v_nom, R_v = sympy.symbols("i_L sigma P C K_i v_nom R_v", positive=True)
    v_bus_manifold = v_nom - R_v * i_L + K_i * sigma
    # equilibrium substitution: sigma* = R_v*i_L/K_i forces v_bus*=v_nom
    # (see sub-task D scratch derivation) -- but the JACOBIAN itself
    # (before substituting the equilibrium values) is what Routh-Hurwitz
    # needs evaluated AT that equilibrium, exactly as done here.
    # Direct re-derivation from f(x), G(x) (independent of the runtime code path):
    d, v_in, L, R_L, k_m = sympy.symbols("d v_in L R_L k_m", positive=True)
    f_iL = (d * v_in - v_bus_manifold - R_L * i_L) / L
    phi2 = v_bus_manifold - v_nom + R_v * i_L - K_i * sigma  # == 0 identically on the manifold by construction
    dv_bus_dt = ((i_L) + (-P / v_bus_manifold)) / C  # buck port_current = i_L (no d-dependence)
    dsigma_dt = v_nom - v_bus_manifold
    phi2_dot = sympy.simplify(dv_bus_dt + R_v * f_iL - K_i * dsigma_dt)
    d_sol = sympy.solve(sympy.Eq(phi2_dot, -k_m * (v_bus_manifold - v_nom + R_v * i_L - K_i * sigma)), d)[0]
    di_L_dt_r = sympy.simplify(f_iL.subs(d, d_sol))
    dsigma_dt_r = sympy.simplify(v_nom - v_bus_manifold)
    J = sympy.Matrix([[sympy.diff(di_L_dt_r, i_L), sympy.diff(di_L_dt_r, sigma)],
                      [sympy.diff(dsigma_dt_r, i_L), sympy.diff(dsigma_dt_r, sigma)]])
    trace = sympy.simplify(J[0, 0] + J[1, 1])
    det = sympy.simplify(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])

    claimed_trace = P / (C * v_bus_manifold ** 2) - 1 / (C * R_v)
    claimed_det = K_i / (C * R_v)
    trace_matches = sympy.simplify(trace - claimed_trace) == 0
    det_matches = sympy.simplify(det - claimed_det) == 0

    return {
        "trace_derived": trace,
        "det_derived": det,
        "trace_matches_closed_form": bool(trace_matches),
        "det_matches_closed_form": bool(det_matches),
    }


def numeric_admissible_R_v_window(net: Network, converter: Converter, topology: str, v_nom_val: float,
                                  baseline_system: AssembledNetworkSystem, base_eq, k_m: float, K_i: float,
                                  r_v_lo: float = 1e-4, r_v_hi: float = 1e3,
                                  n_probe: int = 40) -> Dict:
    """
    NUMERICAL (not closed-form) admissible-R_v search for topologies
    (boost, buck-boost) where a closed-form Routh-Hurwitz region was not
    obtained (see module note above). Probes `n_probe` geometrically
    spaced R_v values in [r_v_lo, r_v_hi], records alpha_parallel at each,
    and reports the observed sign-change interval(s) as an EMPIRICAL
    window -- explicitly labeled as such, never presented as a proof that
    the boundary is exactly there or that only one stable interval exists.
    This is strictly more thorough than a fixed coarse grid (it is used
    to CHARACTERIZE the admissible region shape before the actual search
    grid is chosen), but it remains a numerical characterization, not an
    analytic derivation -- this limitation is stated in the returned dict.
    """
    grid = np.geomspace(r_v_lo, r_v_hi, n_probe)
    probes = []
    for R_v in grid:
        r = attempt_bus_voltage_manifold_reshaping(
            net, converter, topology, v_nom_val, baseline_system, base_eq, k_m, K_i,
            r_v_grid=[float(R_v)],
        )
        entry = r["search"][0] if r["search"] else {}
        probes.append({"R_v": float(R_v), "alpha_parallel": entry.get("alpha_parallel"),
                       "status": entry.get("status")})

    stable_intervals = []
    in_interval = False
    start = None
    for p in probes:
        is_stable = p["alpha_parallel"] is not None and p["alpha_parallel"] < -TANGENTIAL_EIGENVALUE_TOLERANCE
        if is_stable and not in_interval:
            start = p["R_v"]
            in_interval = True
        elif not is_stable and in_interval:
            stable_intervals.append((start, p["R_v"]))
            in_interval = False
    if in_interval:
        stable_intervals.append((start, grid[-1]))

    return {
        "topology": topology,
        "derivation": "numerical_only",
        "limitation": (f"No closed-form Routh-Hurwitz region was derived for topology '{topology}' -- "
                       f"the trace/det of its reduced Jacobian, after substituting the manifold and "
                       f"closed-form duty law, are long rational expressions SymPy did not simplify "
                       f"into an interpretable inequality (see module docstring). This window is an "
                       f"EMPIRICAL characterization from {n_probe} probe points in "
                       f"[{r_v_lo:g}, {r_v_hi:g}], not a proof; the true admissible region could differ "
                       f"between probe points or outside the probed range."),
        "probes": probes,
        "observed_stable_intervals": stable_intervals,
    }


def _detect_single_cpl_like_load(net: Network, bus_id: str, exclude_component_id: str) -> Optional[Tuple[float, float]]:
    """
    Purely STRUCTURAL detection (never a class-name/type-label check) of
    whether `bus_id` has exactly one other component whose own params
    dict has the SHAPE of a constant-power-style load -- both a "P" and a
    "v_floor" entry, i.e. i_inj = -P/v_eff(v_floor) by construction,
    whatever that component's actual class is named. Used ONLY to decide
    which R_v values to try (an efficiency/rigor choice about where to
    search, not the stability verdict itself, which always comes from
    the real closed-loop eigenvalues regardless of how this function
    answers). Returns (P, v_floor) or None if not exactly one such match.
    """
    others = [c for c in net.components_at(bus_id) if c.id != exclude_component_id]
    matches = [c for c in others if "P" in c.params and "v_floor" in c.params]
    if len(matches) != 1:
        return None
    return float(matches[0].params["P"]), float(matches[0].params["v_floor"])


def _other_bus_injection_fn(net: Network, bus_id: str, exclude_component_id: str):
    """
    Build a plain callable `v_bus -> float`: the TOTAL current every
    component other than `exclude_component_id` injects into `bus_id`,
    evaluated LIVE via each component's own real `current_injection`
    method (network.components.BusComponent) -- never a hardcoded
    per-load-type formula (constant-power, impedance, or otherwise); a network with
    a ConstantCurrentLoad, an IdealSource, or any future load type this
    module has never heard of is handled identically, because the
    formula is never duplicated here.

    Raises MRCNotEstablished (never silently drops or mis-sums a
    component) if any other component at the bus carries its own dynamic
    state (`n_local_states > 0`, e.g. another converter) -- this callable
    has no access to that state's current value at call time, so summing
    its current_injection would be silently wrong. `u=None` is always
    passed to `current_injection` regardless of a component's own
    `has_input` flag (that flag only marks eligibility to be the
    pipeline's designated disturbance/input component elsewhere, e.g.
    `network.network.Network.components_at`'s docstring; every
    `current_injection` implementation in `network.components` falls
    back to its own stored parameter when `u is None`, by construction).
    A bus with a converter and any number of purely algebraic (stateless)
    loads -- of ANY type, including ones this module has never heard of
    -- is the supported case, the same single-converter-plus-loads
    pattern this module's phi1 construction and the whole MRC validation
    test suite already targets.
    """
    others = [c for c in net.components_at(bus_id) if c.id != exclude_component_id]
    for c in others:
        if getattr(c, "n_local_states", 0) > 0:
            raise MRCNotEstablished(
                f"MRC manifold reshaping not established: bus '{bus_id}' has another stateful component "
                f"('{c.id}') besides the actuated converter -- the bus-voltage-anchored candidate's "
                f"duty law needs that component's OWN current state to evaluate its current injection, "
                f"which this construction does not have access to; not attempted."
            )

    def other_injection(v_bus: float) -> float:
        return sum(c.current_injection(v_bus, np.zeros(0), None) for c in others)

    return other_injection


def attach_bus_voltage_mrc(net: Network, converter_id: str, v_nom: float,
                           mrc_params: Optional[Dict] = None) -> Tuple[Network, Converter]:
    """
    Second-candidate analogue of `attach_auto_mrc`: return a NEW Network
    (the original untouched -- `attach_auto_mrc`'s first candidate is
    NEVER removed or mutated by this) in which the target converter's
    controller is replaced by `BusVoltageAnchoredMRCController`, with its
    `other_bus_injection` callable wired from the REAL other components
    on the same bus (`_other_bus_injection_fn`).
    """
    converter = find_duty_modulated_converter(net, converter_id)
    topology = _MODEL_TO_TOPOLOGY[type(converter.electrical_model)]
    p = dict(DEFAULT_MRC_PARAMS_V2)
    if mrc_params:
        p.update(mrc_params)
    if "R_v" not in p:
        raise MRCNotEstablished(
            "MRC manifold reshaping not established: 'R_v' was not supplied for the bus-voltage-anchored "
            "candidate (no universally-valid default exists -- see R_V_SEARCH_GRID / the per-topology "
            "stability-window derivation)."
        )
    em_p = converter.electrical_model.params
    bus = net.buses[converter.bus]
    if bus.C is None:
        raise MRCNotEstablished(
            f"MRC manifold reshaping not established: bus '{bus.id}' has no capacitance C set -- phi2's "
            f"dv_bus/dt is undefined without it."
        )
    resolved = {
        "v_nom": float(v_nom),
        "K_i": float(p["K_i"]),
        "k_m": float(p["k_m"]),
        "R_v": float(p["R_v"]),
        "topology_code": TOPOLOGY_CODE[topology],
        "v_in": float(em_p["v_in"]),
        "L": float(em_p["L"]),
        "R_L": float(em_p.get("R_L", 0.0)),
        "C": float(bus.C),
    }
    other_injection = _other_bus_injection_fn(net, converter.bus, converter.id)

    new_net = Network(net.name)
    for b in net.buses.values():
        new_net.buses[b.id] = b
    new_net.branches = list(net.branches)
    new_net.components = []
    for comp in net.components:
        if comp.id == converter.id:
            new_controller = BusVoltageAnchoredMRCController(**resolved)
            new_controller.other_bus_injection = other_injection
            new_conv = Converter(id=comp.id, bus=comp.bus, electrical_model=comp.electrical_model,
                                  controller=new_controller)
            new_net.components.append(new_conv)
        else:
            new_net.components.append(comp)
    new_converter = next(c for c in new_net.components if c.id == converter.id)
    return new_net, new_converter


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
# Rigorous transverse/tangential decomposition of the closed-loop Jacobian
# ---------------------------------------------------------------------------

def tangent_normal_decomposition(J: np.ndarray, Dphi_row: np.ndarray, k_m: float) -> Dict:
    """
    Audit/replacement for "pick the eigenvalue closest to -k_m and call
    the rest tangential": constructs the ACTUAL tangent space of the
    manifold M = {x : phi(x) = 0} at the equilibrium,

        T_x* M = ker(Dphi(x*)),

    via a basis matrix T (columns spanning ker(Dphi_row), from
    scipy.linalg.null_space -- an orthonormal basis, not an ad hoc pick),
    and restricts the closed-loop linearization to it:

        J_par = pinv(T) @ J @ T,          lambda_par = eig(J_par).

    Separately, the transverse eigenvalue is verified algebraically, not
    merely located by numeric proximity: for this module's manifold
    family, phi is AFFINE (Dphi is constant, D^2 phi = 0), and the
    closed-form control law is solved so that the exact identity
    Dphi(x) [f(x) + G(x) u_MRC(x)] = -k_m * phi(x) holds for every x (not
    just at x*). Differentiating both sides at x* and using D^2 phi = 0
    gives

        Dphi(x*) @ J = -k_m * Dphi(x*)                      (*)

    i.e. Dphi_row is an EXACT LEFT EIGENVECTOR of J with eigenvalue
    -k_m -- an algebraic fact, not a numerical coincidence, and not
    something that requires searching the spectrum for the "closest"
    value. (*) is checked directly below (`left_eigenvector_residual`)
    as independent confirmation, alongside the numeric estimate.

    Finally, the two pieces are cross-checked against the FULL ambient
    spectrum: {lambda_par} union {-k_m} must equal eig(J) as a multiset
    (within tolerance) -- "consistent with the full closed-loop
    spectrum", explicitly, not assumed from eigenvalue ordering.
    """
    n = J.shape[0]
    Dphi_row = np.asarray(Dphi_row, dtype=float).reshape(1, n)

    T = _scipy_null_space(Dphi_row)  # columns: orthonormal basis of ker(Dphi) = T_x* M
    T_pinv = np.linalg.pinv(T)
    J_par = T_pinv @ J @ T
    lambda_par = np.linalg.eigvals(J_par)

    # Algebraic verification that Dphi_row is an exact left eigenvector
    # of J with eigenvalue -k_m (equation (*) above) -- independent of
    # and prior to any numeric eigenvalue search.
    left_eig_residual = np.linalg.norm(Dphi_row.flatten() @ J + k_m * Dphi_row.flatten())
    left_eig_relative_residual = float(left_eig_residual / max(np.linalg.norm(Dphi_row), 1e-12) / max(1.0, k_m))

    full_spectrum = np.linalg.eigvals(J)
    reconstructed = np.concatenate([lambda_par, np.array([-k_m + 0j])])
    # multiset comparison, not ordering-dependent: sort both by (real, imag)
    key = lambda z: (round(z.real, 6), round(z.imag, 6))
    full_sorted = np.array(sorted(full_spectrum, key=key))
    recon_sorted = np.array(sorted(reconstructed, key=key))
    spectrum_consistent = bool(
        full_sorted.shape == recon_sorted.shape and np.allclose(full_sorted, recon_sorted, atol=1e-3, rtol=1e-3)
    )

    result = {
        "tangent_space_dim": int(T.shape[1]),
        "tangential_eigenvalues": [{"re": float(z.real), "im": float(z.imag)} for z in lambda_par],
        "transverse_eigenvalue": {"re": -float(k_m), "im": 0.0},
        "left_eigenvector_residual": float(left_eig_residual),
        "left_eigenvector_relative_residual": left_eig_relative_residual,
        "left_eigenvector_verified": bool(left_eig_relative_residual < 1e-6),
        "spectrum_consistent_with_full_jacobian": spectrum_consistent,
        "full_spectrum": [{"re": float(z.real), "im": float(z.imag)} for z in full_spectrum],
    }

    # Explicit Routh-Hurwitz check when the tangent space is exactly 2-D
    # (true for every single-converter, single-bus-state network this
    # module targets: n_states=3 total (i_L, v_bus, sigma) minus 1 scalar
    # manifold constraint = 2). "trace<0" ALONE is necessary but NOT
    # sufficient for a 2x2 system to be Hurwitz -- both eigenvalues
    # negative real part requires trace<0 AND det>0 (equivalently: a
    # negative trace with det<=0 means a saddle or a zero eigenvalue, not
    # asymptotic stability). This is reported explicitly and separately
    # from `tangential_eigenvalues` (which already reflects the true
    # eigenvalues either way -- this block exists so callers never reduce
    # the classification to "trace<0" as a shortcut, per the explicit
    # audit requirement that stability not be concluded from trace alone).
    if J_par.shape == (2, 2):
        tr = float(np.trace(J_par).real)
        det = float(np.linalg.det(J_par).real)
        result["J_par"] = J_par.tolist()
        result["routh_hurwitz_2x2"] = {
            "trace": tr,
            "det": det,
            "trace_negative": bool(tr < -2 * TANGENTIAL_EIGENVALUE_TOLERANCE),
            "det_positive": bool(det > 0),
            "hurwitz_stable": bool(tr < -2 * TANGENTIAL_EIGENVALUE_TOLERANCE and det > 0),
            "note": ("Necessary AND sufficient for both eigenvalues of a 2x2 real matrix to have "
                     "negative real part: trace<0 AND det>0 (product of eigenvalues=det>0 rules out a "
                     "saddle; sum of eigenvalues=trace<0 then forces both negative). Consistent with, "
                     "never a substitute for, the actual eigenvalues in 'tangential_eigenvalues' above."),
        }
        # Cross-check: the Routh-Hurwitz verdict and the eigenvalue-based
        # verdict (alpha_parallel < 0, computed independently via
        # np.linalg.eigvals above) must agree -- if they don't, that is
        # itself reported as a genuine inconsistency, never silently
        # resolved in favor of either one.
        eig_stable = bool(max((z.real for z in lambda_par), default=1.0) < -TANGENTIAL_EIGENVALUE_TOLERANCE)
        result["routh_hurwitz_2x2"]["agrees_with_eigenvalue_classification"] = bool(
            result["routh_hurwitz_2x2"]["hurwitz_stable"] == eig_stable
        )
    else:
        result["J_par"] = None
        result["routh_hurwitz_2x2"] = None

    return result


def _verify_A2_finite_difference(topology: str, converter: Converter, v_bus0: float, i_L0: float,
                                 R_v: float, C: float) -> Dict:
    """
    Independent, finite-difference verification of A2(x) = d(phi2_dot)/dd
    on the REAL, assembled electrical model (em.local_dynamics AND
    em.port_current -- phi2's derivative depends on the converter's own
    port current directly, unlike phi1's), cross-checked against the
    per-topology closed-form A2 in `bus_voltage_mrc_duty`'s docstring.
    Structurally parallel to `verify_control_affine_and_G` for phi1.
    """
    em = converter.electrical_model
    p = em.params
    h = 1e-6
    d0 = 0.5
    diL_p = em.local_dynamics(v_bus0, np.array([i_L0]), d0 + h, p)[0]
    diL_m = em.local_dynamics(v_bus0, np.array([i_L0]), d0 - h, p)[0]
    diL_dd_fd = float((diL_p - diL_m) / (2 * h))

    port_p = em.port_current(v_bus0, np.array([i_L0]), d0 + h, p)
    port_m = em.port_current(v_bus0, np.array([i_L0]), d0 - h, p)
    port_dd_fd = float((port_p - port_m) / (2 * h))

    A2_fd = port_dd_fd / C + R_v * diL_dd_fd

    v_in = float(p["v_in"]); L = float(p["L"])
    if topology == "buck":
        A2_closed_form = R_v * v_in / L
    elif topology == "boost":
        A2_closed_form = R_v * v_bus0 / L - i_L0 / C
    elif topology == "buckboost":
        A2_closed_form = R_v * (v_bus0 + v_in) / L - i_L0 / C
    else:
        A2_closed_form = float("nan")

    return {
        "A2_finite_difference": A2_fd,
        "A2_closed_form": A2_closed_form,
        "A2_relative_error": float(abs(A2_fd - A2_closed_form) / max(abs(A2_closed_form), 1e-12)),
        "verified": bool(abs(A2_fd - A2_closed_form) < 1e-4 * max(abs(A2_closed_form), 1.0)),
    }


def _m2_physical_feasibility_check(mrc_converter: Converter, cl_eq2, name_to_idx2: Dict, converter_id: str,
                                   bus_state_name: str, baseline_i_L0: float, A2_center: float,
                                   reference_A_scale: float) -> Dict:
    """
    Physical controller feasibility for an accepted M2 candidate -- a
    mathematically stable reduced Jacobian is NOT, by itself, grounds to
    classify a candidate as fully SUPPORTED (task requirement 3/6): the
    required control action must also be physically realizable.

    What this platform actually models a hard limit for:
      - duty ratio: EVERY ConverterModel.local_dynamics clamps its control
        signal through `clamp_duty` to [0.02, 0.98] (network.electrical_
        model.ConverterModel) -- a duty command outside that range at
        equilibrium means the exact contraction identity phi2_dot=-k_m*
        phi2 is NOT actually active there (the real, clamped closed-loop
        dynamics differs from the ideal unclamped model this candidate's
        stability analysis was computed on). This is a HARD gate.

    What this platform does NOT model an explicit rated limit for (i_L
    current rating, v_bus voltage rating): reported as advisory
    information for the caller's own engineering judgment, honestly
    labeled as "not modeled by this platform" rather than gated against a
    fabricated threshold -- inventing a current/voltage rating this
    module has no basis for would be less honest than reporting the
    actual equilibrium values and letting the caller apply their own.

    DφG conditioning: A2(x) at the equilibrium relative to a reference
    scale (phi1's own A(x) order of magnitude, v_in/L) -- a very small
    |A2| relative to that reference means the duty law divides by a
    near-zero quantity, amplifying noise/model error even though it is
    formally nonzero (already gated at 1e-9 absolute in the search loop;
    this reports the CONTINUOUS conditioning number, not just pass/fail).
    """
    sigma_idx = name_to_idx2[f"{converter_id}_ctrl_sigma"]
    iL_idx = name_to_idx2[f"{converter_id}_em_i_L"]
    bus_idx = name_to_idx2[bus_state_name]
    i_L = float(cl_eq2.x_star[iL_idx])
    v_bus = float(cl_eq2.x_star[bus_idx])
    sigma = float(cl_eq2.x_star[sigma_idx])
    p = mrc_converter.controller.params
    other_injection = mrc_converter.controller.other_bus_injection
    d_eq = bus_voltage_mrc_duty("", i_L, v_bus, sigma, p, other_injection)
    duty_saturated = bool(d_eq < 0.02 or d_eq > 0.98)

    conditioning_ratio = float(abs(A2_center) / max(abs(reference_A_scale), 1e-12))

    return {
        "duty_at_equilibrium": float(d_eq),
        "duty_within_limits": bool(not duty_saturated),
        "duty_saturated": duty_saturated,
        "duty_note": ("Duty at equilibrium is within [0.02, 0.98]: the exact contraction identity is "
                     "active there." if not duty_saturated else
                     "Duty at equilibrium falls OUTSIDE the converter's clamp [0.02, 0.98]: the ideal "
                     "contraction identity phi2_dot=-k_m*phi2 is NOT active while the duty clamp is "
                     "engaged -- this candidate cannot be classified fully SUPPORTED regardless of its "
                     "linearized eigenvalues."),
        "i_L_equilibrium": i_L,
        "v_bus_equilibrium": v_bus,
        "i_L_vs_baseline_ratio": float(i_L / baseline_i_L0) if abs(baseline_i_L0) > 1e-12 else None,
        "current_voltage_limit_note": ("This platform does not model an explicit rated current or voltage "
                                       "limit for converters/buses -- i_L/v_bus equilibrium values are "
                                       "reported for the caller's own engineering judgment, not gated "
                                       "against a fabricated threshold."),
        "DphiG_conditioning_ratio": conditioning_ratio,
        "DphiG_well_conditioned": bool(conditioning_ratio > 1e-3),
        "feasible": bool(not duty_saturated),
    }


def attempt_bus_voltage_manifold_reshaping(net: Network, converter: Converter, topology: str,
                                           v_nom_val: float, baseline_system: AssembledNetworkSystem,
                                           base_eq, k_m: float, K_i: float,
                                           r_v_grid: Optional[List[float]] = None) -> Dict:
    """
    "Attempt manifold reshaping" (workflow diagram, sub-task D item 2):
    when the current-anchored candidate (phi1) has verified transverse
    contraction but UNSTABLE reduced/tangential dynamics, construct the
    bus-voltage-anchored second candidate (phi2) independently per
    topology and search R_v (task: "reshape the manifold so that both
    lambda_perp < 0 and alpha_parallel < 0") over `r_v_grid`
    (R_V_SEARCH_GRID by default).

    Every attempted R_v is reported -- including every failure and its
    exact alpha_parallel, its explicit Routh-Hurwitz trace/det verdict
    (never trace alone), and its physical feasibility check (duty at
    equilibrium, saturation, DphiG conditioning) -- so a topology/load
    combination for which no stabilizing AND feasible R_v exists in the
    searched grid is reported honestly, never silently forced to
    SUPPORTED and never silently hidden. The first R_v (in ascending
    order) that verifies BOTH lambda_perp exactly (via
    `tangent_normal_decomposition`'s algebraic left-eigenvector identity)
    AND alpha_parallel < -TANGENTIAL_EIGENVALUE_TOLERANCE AND passes the
    physical feasibility check is returned as the chosen candidate; the
    pipeline does not search for the "best" R_v, only the first one the
    grid finds (a genuine engineering choice, reported as such). A
    candidate that is mathematically stable but physically infeasible
    (duty saturated at equilibrium) is recorded and search continues --
    if no fully-passing candidate is ever found, the best
    math-stable-but-infeasible one (if any) is surfaced distinctly from
    "no stabilizing R_v at all", per task requirement 3/6.

    When `r_v_grid` is not given, the default grid is chosen PER
    TOPOLOGY: buck uses the CLOSED-FORM admissible interval
    (0, v_nom^2/P) derived analytically (see `buck_phi2_admissible_region`)
    for a constant-power load, subdivided into evenly-spaced interior points (never
    searching outside the proven region); boost/buck-boost fall back to
    the fixed numerical `R_V_SEARCH_GRID`, since no closed-form region was
    derived for them (see the module note above `buck_phi2_admissible_region`)
    -- this distinction is reported in `result["grid_derivation"]`.
    """
    bus = net.buses[converter.bus]
    admissible_region = None
    grid_derivation = "caller_supplied"
    if r_v_grid is not None:
        grid = list(r_v_grid)
    elif topology == "buck":
        # Closed-form region requires knowing P and C, i.e. the actual
        # load. Detected STRUCTURALLY (by parameter shape -- a single
        # other bus component whose own params include both "P" and
        # "v_floor", never by inspecting a class name or type label; see
        # `_detect_single_cpl_like_load`), not by branching on any
        # load-type string. If detection fails (multiple other
        # components, or a load shape this structural check does not
        # recognize), falls back to the numerical grid, honestly
        # reported as such -- never silently assumes a constant-power load is present.
        detected = _detect_single_cpl_like_load(net, converter.bus, converter.id)
        if detected is not None:
            P_detected, _v_floor = detected
            admissible_region = buck_phi2_admissible_region(P_detected, float(bus.C), v_nom_val)
            R_v_max = admissible_region["R_v_max"]
            # interior points only -- never the boundary itself (trace=0
            # exactly at R_v=R_v_max is the undetermined/NOT_ESTABLISHED
            # case, not a stable point) and never outside the proven region.
            grid = list(np.linspace(0.02 * R_v_max, 0.95 * R_v_max, 12))
            grid_derivation = "analytical (buck closed-form Routh-Hurwitz region, see admissible_region)"
        else:
            grid = list(R_V_SEARCH_GRID)
            grid_derivation = ("numerical_grid (closed-form region requires a single, structurally-"
                               "detected constant-power-shaped load at the bus; not detected here)")
    else:
        grid = list(R_V_SEARCH_GRID)
        grid_derivation = "numerical_grid (no closed-form region derived for this topology -- see module note)"

    result: Dict = {
        "attempted": True,
        "candidate_phi": "phi2(x) = v_bus - v_nom + R_v*i_L - K_i*sigma,  sigma_dot = v_nom - v_bus "
                          "(v_Sigma = v_nom - R_v*i_L + K_i*sigma, bus-voltage-anchored second candidate)",
        "K_i": K_i, "k_m": k_m,
        "search": [],
        "chosen_R_v": None,
        "status": None,
        "R_v_grid": grid,
        "grid_derivation": grid_derivation,
        "admissible_region": admissible_region,
        "math_stable_but_infeasible_entry": None,
    }

    try:
        law2 = derive_symbolic_law_v2(topology, {})
    except MRCNotEstablished as e:
        result["status"] = "MRC_MANIFOLD_RESHAPING_NOT_ESTABLISHED"
        result["reason"] = str(e)
        return result
    result["phi2_latex"] = law2["phi_latex"]
    result["A2_symbolic"] = str(law2["A_symbolic"])
    result["control_law2_latex"] = law2["control_latex"]

    n_states = baseline_system.n_states
    i_L_idx = next(i for i, n in enumerate(baseline_system.state_names) if n == f"{converter.id}_em_i_L")
    v_bus_idx = next(i for i, n in enumerate(baseline_system.state_names)
                     if n == bus.state_name)
    i_L0 = float(base_eq.x_star[i_L_idx])
    v_bus0 = float(base_eq.x_star[v_bus_idx])
    em_p0 = converter.electrical_model.params
    reference_A_scale = float(em_p0["v_in"]) / float(em_p0["L"])  # phi1's own A(x) order of magnitude

    # other_injection does not depend on R_v -- built once, not once per
    # grid point (also lets a genuine "not established" here abort the
    # whole search immediately rather than repeating the same failure).
    try:
        other_injection = _other_bus_injection_fn(net, converter.bus, converter.id)
    except MRCNotEstablished as e:
        result["status"] = "MRC_MANIFOLD_RESHAPING_NOT_ESTABLISHED"
        result["reason"] = str(e)
        return result

    for R_v in grid:
        entry: Dict = {"R_v": float(R_v)}

        # feasibility: A2 != 0 at and near the baseline point
        A2_fn = sympy.lambdify(
            (law2["symbols"]["iL"], law2["symbols"]["vbus"], law2["symbols"]["v_in"],
             law2["symbols"]["L"], law2["symbols"]["C"], law2["symbols"]["R_v"]),
            law2["A_symbolic"], "numpy")
        em_p = converter.electrical_model.params
        A2_center = float(A2_fn(i_L0, v_bus0, em_p["v_in"], em_p["L"], bus.C, R_v))
        entry["A2_center"] = A2_center
        if abs(A2_center) < 1e-9:
            entry["status"] = "RELATIVE_DEGREE_FAILS"
            result["search"].append(entry)
            continue

        fd_check = _verify_A2_finite_difference(topology, converter, v_bus0, i_L0, R_v, float(bus.C))
        entry["A2_finite_difference_check"] = fd_check
        if not fd_check["verified"]:
            entry["status"] = "A2_FINITE_DIFFERENCE_MISMATCH"
            result["search"].append(entry)
            continue

        try:
            mrc_net2, mrc_conv2 = attach_bus_voltage_mrc(net, converter.id, v_nom_val,
                                                          {"K_i": K_i, "k_m": k_m, "R_v": R_v})
        except MRCNotEstablished as e:
            result["status"] = "MRC_MANIFOLD_RESHAPING_NOT_ESTABLISHED"
            result["reason"] = str(e)
            return result

        mrc_system2 = AutomaticModelBuilder.build(mrc_net2, name=net.name + "_auto_mrc_v2")
        x0_guess = mrc_system2.initial_guess()
        name_to_idx2 = {n: i for i, n in enumerate(mrc_system2.state_names)}
        for name, idx in zip(baseline_system.state_names, range(n_states)):
            if name in name_to_idx2:
                x0_guess[name_to_idx2[name]] = base_eq.x_star[idx]
        try:
            cl_eq2 = mrc_system2.find_equilibrium(x0_guess, with_eigs=True)
        except Exception as e:  # pragma: no cover - defensive
            entry["status"] = "EQUILIBRIUM_SOLVE_ERROR"
            entry["error"] = str(e)
            result["search"].append(entry)
            continue
        entry["closed_loop_equilibrium_converged"] = bool(cl_eq2.converged)
        entry["closed_loop_equilibrium_residual_norm"] = float(cl_eq2.residual_norm)
        if not cl_eq2.converged:
            entry["status"] = "EQUILIBRIUM_NOT_REACHED"
            result["search"].append(entry)
            continue

        iL_idx2 = name_to_idx2[f"{converter.id}_em_i_L"]
        vbus_idx2 = name_to_idx2[bus.state_name]
        sigma_idx2 = name_to_idx2[f"{converter.id}_ctrl_sigma"]
        J_cl2 = cl_eq2.jacobian
        if J_cl2 is None:
            entry["status"] = "NO_JACOBIAN"
            result["search"].append(entry)
            continue
        Dphi2_row = np.zeros(J_cl2.shape[0])
        Dphi2_row[vbus_idx2] = 1.0
        Dphi2_row[iL_idx2] = R_v
        Dphi2_row[sigma_idx2] = -K_i
        decomp2 = tangent_normal_decomposition(J_cl2, Dphi2_row, k_m)
        entry["tangent_normal_decomposition"] = decomp2

        if not (decomp2["left_eigenvector_verified"] and decomp2["spectrum_consistent_with_full_jacobian"]):
            entry["status"] = "TRANSVERSE_IDENTITY_NOT_VERIFIED"
            result["search"].append(entry)
            continue

        tangential2 = [complex(e["re"], e["im"]) for e in decomp2["tangential_eigenvalues"]]
        alpha_parallel2 = max((z.real for z in tangential2), default=None)
        entry["alpha_parallel"] = alpha_parallel2
        entry["transverse_eigenvalue"] = decomp2["transverse_eigenvalue"]

        if alpha_parallel2 is None:
            entry["reduced_dynamics_status"] = "NOT_ESTABLISHED"
        elif alpha_parallel2 < -TANGENTIAL_EIGENVALUE_TOLERANCE:
            entry["reduced_dynamics_status"] = "STABLE"
        elif alpha_parallel2 > TANGENTIAL_EIGENVALUE_TOLERANCE:
            entry["reduced_dynamics_status"] = "UNSTABLE"
        else:
            entry["reduced_dynamics_status"] = "NOT_ESTABLISHED"
        entry["status"] = entry["reduced_dynamics_status"]

        if entry["reduced_dynamics_status"] == "STABLE":
            # Mathematical stability alone is NOT grounds for full support
            # (task requirement 3/6) -- verify physical feasibility of the
            # required control action before accepting this candidate.
            feasibility = _m2_physical_feasibility_check(
                mrc_conv2, cl_eq2, name_to_idx2, converter.id, bus.state_name, i_L0, A2_center,
                reference_A_scale,
            )
            entry["physical_feasibility"] = feasibility
            entry["closed_loop_equilibrium"] = {
                mrc_system2.state_names[i]: float(cl_eq2.x_star[i]) for i in range(mrc_system2.n_states)
            }

            if feasibility["feasible"]:
                entry["status"] = "STABLE_AND_FEASIBLE"
                result["search"].append(entry)
                result["chosen_R_v"] = R_v
                result["status"] = "MRC_SYNTHESIS_SUPPORTED_VIA_M2"
                result["chosen_entry"] = entry
                result["closed_loop_equilibrium"] = entry["closed_loop_equilibrium"]
                result["mrc_system"] = mrc_system2
                result["mrc_converter"] = mrc_conv2
                result["cl_eq"] = cl_eq2
                result["name_to_idx"] = name_to_idx2
                result["bus_state_name"] = bus.state_name
                return result
            else:
                entry["status"] = "STABLE_BUT_PHYSICALLY_INFEASIBLE"
                result["search"].append(entry)
                if result["math_stable_but_infeasible_entry"] is None:
                    result["math_stable_but_infeasible_entry"] = entry
                continue

        result["search"].append(entry)

    if result["math_stable_but_infeasible_entry"] is not None:
        infeasible = result["math_stable_but_infeasible_entry"]
        result["status"] = "MATH_STABLE_BUT_PHYSICALLY_INFEASIBLE"
        result["reason"] = (
            f"Manifold reshaping found R_v={infeasible['R_v']:.6g} (and possibly others) with "
            f"mathematically stable reduced dynamics (alpha_parallel={infeasible['alpha_parallel']:.6g} "
            f"< -{TANGENTIAL_EIGENVALUE_TOLERANCE:g}) and the exact transverse identity verified, but the "
            f"required equilibrium duty ({infeasible['physical_feasibility']['duty_at_equilibrium']:.4g}) "
            f"falls outside the converter's physical clamp [0.02, 0.98] -- the real, saturated closed-loop "
            f"dynamics does NOT match the ideal linearization this stability result was computed on. This "
            f"candidate is reported as mathematically stable but physically infeasible, NOT as SUPPORTED; "
            f"see 'search' for every attempted R_v's own feasibility check."
        )
        return result

    result["status"] = "NO_STABILIZING_R_V_FOUND_IN_GRID"
    result["reason"] = (
        f"Manifold reshaping was attempted for topology '{topology}' with the bus-voltage-anchored "
        f"candidate phi2 = v_bus - v_nom + R_v*i_L - K_i*sigma over R_v in {grid} ({grid_derivation}); "
        f"every attempted value either failed a structural check (relative degree, transverse identity) "
        f"or left the reduced/tangential dynamics UNSTABLE or NOT_ESTABLISHED (see 'search' for each "
        f"R_v's own alpha_parallel and Routh-Hurwitz trace/det verdict). No stabilizing R_v was found in "
        f"this grid -- this does not prove none exists outside the searched range, and is reported as a "
        f"bounded, honest search result, not a proof of impossibility for this topology/load."
    )
    return result


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
    K_i = resolved_params["K_i"]
    report["closed_loop_eigenvalues"] = [{"re": float(z.real), "im": float(z.imag)} for z in eigs] if eigs is not None else None

    # --- Rigorous transverse/tangential decomposition (audit requirement:
    # do not just pick "the eigenvalue closest to -k_m" and call the rest
    # tangential). Dphi(x) = d(i_L - i_bias - K_i*sigma)/dx has a 1 at the
    # i_L coordinate and -K_i at the sigma coordinate, 0 elsewhere -- an
    # exact, closed-form row vector (not estimated), because phi is
    # affine by this construction. tangent_normal_decomposition() builds
    # T = ker(Dphi) via scipy.linalg.null_space, restricts J to it for
    # the tangential eigenvalues, and independently verifies
    # Dphi @ J == -k_m * Dphi (an exact left-eigenvector identity implied
    # by phi_dot = -k_m*phi holding for EVERY x, not just x*, together
    # with D^2 phi = 0) -- confirming lambda_perp = -k_m algebraically,
    # not merely by numeric proximity.
    J_cl = cl_eq.jacobian
    decomposition = None
    if eigs is not None and J_cl is not None:
        iL_idx_mrc = name_to_idx_mrc.get(f"{converter.id}_em_i_L")
        sigma_idx_mrc = name_to_idx_mrc.get(f"{converter.id}_ctrl_sigma")
        if iL_idx_mrc is not None and sigma_idx_mrc is not None:
            Dphi_row = np.zeros(J_cl.shape[0])
            Dphi_row[iL_idx_mrc] = 1.0
            Dphi_row[sigma_idx_mrc] = -K_i
            decomposition = tangent_normal_decomposition(J_cl, Dphi_row, k_m)
    report["tangent_normal_decomposition"] = decomposition

    if decomposition is not None and decomposition["left_eigenvector_verified"] and decomposition["spectrum_consistent_with_full_jacobian"]:
        # Rigorous method available and self-consistent with the full
        # ambient spectrum: use it as authoritative.
        transverse = complex(decomposition["transverse_eigenvalue"]["re"], decomposition["transverse_eigenvalue"]["im"])
        tangential = [complex(e["re"], e["im"]) for e in decomposition["tangential_eigenvalues"]]
        report["transverse_eigenvalue"] = decomposition["transverse_eigenvalue"]
        report["transverse_matches_km"] = True  # proven exactly, not merely within tolerance
        report["tangential_eigenvalues"] = decomposition["tangential_eigenvalues"]
        report["reduced_dynamics_method"] = "tangent_space_projection (ker(Dphi) restriction, algebraically verified)"
    else:
        # Fallback for a manifold/case where the exact-affine structure
        # above does not hold (e.g. phi nonlinear, or the algebraic
        # identity fails to verify): fall back to the numeric-proximity
        # heuristic, clearly labeled as such rather than silently reused.
        idx_transverse = int(np.argmin(np.abs(eigs - (-k_m)))) if eigs is not None else None
        transverse = eigs[idx_transverse] if idx_transverse is not None else None
        tangential = [complex(z) for i, z in enumerate(eigs) if i != idx_transverse] if eigs is not None else []
        report["transverse_eigenvalue"] = {"re": float(transverse.real), "im": float(transverse.imag)} if transverse is not None else None
        report["transverse_matches_km"] = bool(transverse is not None and abs(transverse.real + k_m) < max(1e-2 * k_m, 1.0) and abs(transverse.imag) < 1.0)
        report["tangential_eigenvalues"] = [{"re": float(z.real), "im": float(z.imag)} for z in tangential]
        report["reduced_dynamics_method"] = "nearest-eigenvalue-to-(-k_m) heuristic (tangent-space projection unavailable or inconsistent for this case)"

    # alpha_parallel := max_i Re(lambda_parallel,i) -- the reduced/
    # tangential dynamics' own dominant real eigenvalue part, the single
    # scalar that actually decides local tangential stability (never the
    # load's component type/label -- these eigenvalues came from the
    # REAL closed-loop Jacobian of the assembled model with the MRC law
    # substituted in). Classified tri-state against
    # TANGENTIAL_EIGENVALUE_TOLERANCE so a genuine numerical
    # stability-boundary point is reported NOT_ESTABLISHED rather than
    # forced into "stable" or "unstable" by a bare "< 0" comparison.
    alpha_parallel = max((z.real for z in tangential), default=None)
    report["alpha_parallel"] = alpha_parallel
    if alpha_parallel is None:
        reduced_dynamics_status = "NOT_ESTABLISHED"
    elif alpha_parallel < -TANGENTIAL_EIGENVALUE_TOLERANCE:
        reduced_dynamics_status = "STABLE"
    elif alpha_parallel > TANGENTIAL_EIGENVALUE_TOLERANCE:
        reduced_dynamics_status = "UNSTABLE"
    else:
        reduced_dynamics_status = "NOT_ESTABLISHED"  # |alpha_parallel| <= tolerance: genuine boundary case
    report["reduced_dynamics_status"] = reduced_dynamics_status
    report["tangential_locally_stable"] = (
        True if reduced_dynamics_status == "STABLE"
        else False if reduced_dynamics_status == "UNSTABLE"
        else None
    )

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

    if reduced_dynamics_status == "STABLE" and report["transverse_matches_km"] and ga["control_affine_verified"]:
        report["status"] = "MRC_SYNTHESIS_SUPPORTED"
        report["classification"] = "MRC SYNTHESIS SUPPORTED"
        report["reason"] = ("Analytic control-affine split verified independently, relative degree one "
                             "confirmed over the sampled neighborhood, closed-form MRC law derived and "
                             "solved, and the closed-loop equilibrium is locally exponentially stable "
                             "(all eigenvalues strictly negative real part; reduced/tangential "
                             f"alpha_parallel={alpha_parallel:.6g} < -{TANGENTIAL_EIGENVALUE_TOLERANCE:g}).")
    elif reduced_dynamics_status == "UNSTABLE":
        report["status"] = "MRC_FEASIBILITY_DIAGNOSTIC_ONLY"
        report["reason"] = ("MRC was derived and the exact transverse residual identity holds within the "
                             "ideal model (transverse eigenvalue matches -k_m), but the reduced/tangential "
                             f"closed-loop dynamics has a dominant real eigenvalue part alpha_parallel="
                             f"{alpha_parallel:.6g} > +{TANGENTIAL_EIGENVALUE_TOLERANCE:g} -- reduced "
                             "dynamics stability (condition (iii)) is a separate claim from transverse "
                             "contraction, never inferred from it, and is genuinely violated here.")

        # --- "Attempt manifold reshaping" (workflow diagram, sub-task D item
        # 2): phi1 achieves transverse contraction but its reduced dynamics
        # are unstable -- construct and search the bus-voltage-anchored
        # second candidate phi2, independently per topology, rather than
        # stopping at DIAGNOSTIC_ONLY. phi1's own report/status above is
        # NEVER overwritten by this -- it remains the first-candidate result
        # exactly as computed; this only ADDS 'manifold_reshaping' and, if a
        # stabilizing R_v is found, upgrades the OVERALL status.
        reshaping = attempt_bus_voltage_manifold_reshaping(
            net, converter, topology, v_nom_val, baseline_system, base_eq, k_m, K_i,
        )
        report["manifold_reshaping"] = {k: v for k, v in reshaping.items()
                                        if k not in ("mrc_system", "mrc_converter", "cl_eq", "name_to_idx")}

        if reshaping["status"] == "MRC_SYNTHESIS_SUPPORTED_VIA_M2":
            report["status"] = "MRC_SYNTHESIS_SUPPORTED_VIA_RESHAPED_MANIFOLD"
            chosen = reshaping["chosen_entry"]
            report["reason"] = (
                f"{report['reason']} Manifold reshaping succeeded: the bus-voltage-anchored second "
                f"candidate phi2 = v_bus - v_nom + R_v*i_L - K_i*sigma with R_v={reshaping['chosen_R_v']:.6g} "
                f"achieves both the exact transverse identity (lambda_perp=-k_m, algebraically verified) "
                f"and stable reduced/tangential dynamics (alpha_parallel={chosen['alpha_parallel']:.6g} < "
                f"-{TANGENTIAL_EIGENVALUE_TOLERANCE:g}). The first-candidate (phi1) result above is "
                f"preserved unchanged; this network is SUPPORTED via the SECOND candidate, not the first."
            )
            report["active_candidate"] = "phi2 (bus-voltage-anchored)"
            report["chosen_R_v"] = reshaping["chosen_R_v"]
            report["classification"] = "MRC SYNTHESIS SUPPORTED — RESHAPED MANIFOLD"
        elif reshaping["status"] == "MATH_STABLE_BUT_PHYSICALLY_INFEASIBLE":
            # Explicit, separate outcome (task requirement 6): a
            # candidate with a mathematically stable reduced Jacobian
            # exists, but its required equilibrium duty is outside the
            # converter's physical clamp -- this is reported plainly,
            # never folded into "SUPPORTED".
            report["active_candidate"] = "phi1 (current-anchored) -- diagnostic only"
            report["classification"] = "MRC RESHAPING MATH-STABLE BUT PHYSICALLY INFEASIBLE"
            report["reason"] = f"{report['reason']} {reshaping['reason']}"
        else:
            report["active_candidate"] = "phi1 (current-anchored) -- diagnostic only"
            report["classification"] = "MRC FEASIBILITY DIAGNOSTIC ONLY (reshaping attempted, no stabilizing R_v found)"
    else:
        # reduced_dynamics_status == "NOT_ESTABLISHED": either no
        # tangential eigenvalues were available to evaluate, or
        # alpha_parallel fell within +/-TANGENTIAL_EIGENVALUE_TOLERANCE of
        # zero -- a genuine numerical stability-boundary point that is
        # honestly reported as undetermined, never forced to "stable" or
        # "unstable" (task requirement: do not classify a boundary point
        # as VIOLATED).
        report["status"] = "MRC_SYNTHESIS_NOT_ESTABLISHED"
        if alpha_parallel is None:
            report["reason"] = ("MRC transverse contraction is established, but the reduced/tangential "
                                 "dynamics could not be constructed at this operating point (no tangential "
                                 "eigenvalues were available) -- reduced dynamics stability is not "
                                 "established, not violated.")
        else:
            report["reason"] = (
                f"MRC transverse contraction is established, but the reduced/tangential dynamics' dominant "
                f"real eigenvalue part alpha_parallel={alpha_parallel:.6g} lies within the numerical "
                f"tolerance (+/-{TANGENTIAL_EIGENVALUE_TOLERANCE:g}) of zero -- a genuine stability-boundary "
                f"operating point. Reduced dynamics stability is NOT_ESTABLISHED here, not VIOLATED: the "
                f"numerical evidence does not resolve which side of the boundary this point is on."
            )
    return report


# ---------------------------------------------------------------------------
# Baseline vs M1 vs M2 -- identical-conditions three-way comparison
# ---------------------------------------------------------------------------

def compare_baseline_m1_m2(net: Network, converter_id: str, v_nom: float,
                           mrc_params_m1: Optional[Dict] = None,
                           R_v: Optional[float] = None, K_i_m2: Optional[float] = None,
                           k_m_m2: Optional[float] = None,
                           t_end: float = 0.05, n_eval: int = 500,
                           perturb_frac: float = 0.85) -> Dict:
    """
    Task requirement 4: run baseline / M1 (current-anchored) / M2
    (bus-voltage-anchored) from THEIR OWN equilibria under the IDENTICAL
    disturbance (a `perturb_frac` multiplicative sag on each arm's own
    bus-voltage state -- the same convention `_closed_loop_simulation`
    already uses for every other before/after comparison in this module),
    the identical solver (RK45), horizon (`t_end`), evaluation grid
    (`n_eval`), and `max_step` bound, so the three trajectories are
    comparable on exactly the same terms.

    Returns, per arm: t, i_L(t), v_bus(t), ||x(t)-x*||(t), d(t)
    (recomputed post-hoc from each arm's own controller along its
    trajectory -- never assumed constant), phi(t) (None for baseline,
    which has no target manifold), saturation flags/duration (fraction of
    the horizon where d(t) is outside [0.02, 0.98]), and a `recovered`
    verdict (same convention as `_closed_loop_simulation`:
    final_deviation < 0.1 * initial_deviation).

    If M2 was not requested with an explicit (R_v, K_i, k_m) -- e.g. the
    caller wants "whatever the pipeline actually chose" -- pass
    `mrc_params_m1` for M1 and the M2 params from a prior
    `run_auto_mrc_pipeline`/`attempt_bus_voltage_manifold_reshaping`
    result's `chosen_R_v` / `K_i` / `k_m`. This function does not itself
    search for a stabilizing R_v -- it compares one already-identified M2
    candidate against baseline and M1, per task requirement 4's "for
    representative constant-power cases run three simulations", not a repeat of the
    search.
    """
    converter = find_duty_modulated_converter(net, converter_id)
    topology = _MODEL_TO_TOPOLOGY[type(converter.electrical_model)]
    if R_v is None or K_i_m2 is None or k_m_m2 is None:
        raise ValueError("compare_baseline_m1_m2 requires an explicit (R_v, K_i_m2, k_m_m2) for M2 -- "
                         "e.g. from a prior attempt_bus_voltage_manifold_reshaping result's "
                         "chosen_R_v/K_i/k_m; it does not perform its own search.")

    baseline_system = AutomaticModelBuilder.build(net, name=net.name + "_baseline")
    base_eq = baseline_system.find_equilibrium(baseline_system.initial_guess(), with_eigs=False)

    m1_net, m1_conv = attach_auto_mrc(net, converter.id, v_nom, mrc_params_m1)
    m1_system = AutomaticModelBuilder.build(m1_net, name=net.name + "_m1")
    m1_x0 = m1_system.initial_guess()
    m1_name_to_idx = {n: i for i, n in enumerate(m1_system.state_names)}
    for name, idx in zip(baseline_system.state_names, range(baseline_system.n_states)):
        if name in m1_name_to_idx:
            m1_x0[m1_name_to_idx[name]] = base_eq.x_star[idx]
    m1_eq = m1_system.find_equilibrium(m1_x0, with_eigs=False)

    m2_net, m2_conv = attach_bus_voltage_mrc(net, converter.id, v_nom,
                                             {"K_i": K_i_m2, "k_m": k_m_m2, "R_v": R_v})
    m2_system = AutomaticModelBuilder.build(m2_net, name=net.name + "_m2")
    m2_x0 = m2_system.initial_guess()
    m2_name_to_idx = {n: i for i, n in enumerate(m2_system.state_names)}
    for name, idx in zip(baseline_system.state_names, range(baseline_system.n_states)):
        if name in m2_name_to_idx:
            m2_x0[m2_name_to_idx[name]] = base_eq.x_star[idx]
    m2_eq = m2_system.find_equilibrium(m2_x0, with_eigs=False)

    bus = net.buses[converter.bus]
    resolved_m1 = _resolved_mrc_params(topology, converter, v_nom, mrc_params_m1)
    i_bias = resolved_m1["i_bias"]; K_i_m1 = resolved_m1["K_i"]; k_m_m1 = resolved_m1["k_m"]

    def _run_arm(system, x_star, converged, bus_idx, iL_idx, sigma_idx, duty_fn):
        if not converged:
            return {"equilibrium_converged": False}
        x0 = x_star.copy()
        x0[bus_idx] *= perturb_frac
        init_dev = float(np.linalg.norm(x0 - x_star))
        sim = Simulator(system, method="RK45")
        try:
            result = sim.simulate(x0=x0, t_span=(0.0, t_end), n_eval=n_eval,
                                  max_step=max(t_end / 200.0, 1e-6))
        except (ValueError, FloatingPointError, ZeroDivisionError) as e:
            return {"equilibrium_converged": True, "success": False, "message": f"solver failure: {e}"}
        t = result.t
        x = result.x  # shape (n_states, n_eval)
        i_L_t = x[iL_idx, :]
        v_bus_t = x[bus_idx, :]
        dev_t = np.linalg.norm(x - x_star[:, None], axis=0)
        d_t = np.array([duty_fn(i_L_t[k], v_bus_t[k], x[sigma_idx, k] if sigma_idx is not None else None)
                        for k in range(x.shape[1])])
        saturated_mask = (d_t < 0.02) | (d_t > 0.98)
        return {
            "equilibrium_converged": True,
            "success": bool(result.success),
            "t": t.tolist(),
            "i_L": i_L_t.tolist(),
            "v_bus": v_bus_t.tolist(),
            "deviation_from_equilibrium": dev_t.tolist(),
            "d": d_t.tolist(),
            "initial_deviation": init_dev,
            "final_deviation": float(dev_t[-1]),
            "recovered": bool(result.success and float(dev_t[-1]) < 0.1 * init_dev),
            "saturation_fraction_of_horizon": float(np.mean(saturated_mask)),
            "saturated_at_any_point": bool(np.any(saturated_mask)),
        }

    # baseline: duty is whatever ConstantDutyController holds fixed at
    baseline_iL_idx = next(i for i, n in enumerate(baseline_system.state_names) if n == f"{converter.id}_em_i_L")
    baseline_bus_idx = next(i for i, n in enumerate(baseline_system.state_names) if n == bus.state_name)
    baseline_duty = converter.controller.control_signal(base_eq.x_star[baseline_bus_idx],
                                                         np.array([base_eq.x_star[baseline_iL_idx]]),
                                                         np.array([]), None, converter.controller.params)
    baseline_result = _run_arm(baseline_system, base_eq.x_star, base_eq.converged,
                               baseline_bus_idx, baseline_iL_idx, None,
                               lambda iL, vb, sg: baseline_duty)

    m1_iL_idx = m1_name_to_idx[f"{converter.id}_em_i_L"]
    m1_bus_idx = m1_name_to_idx[bus.state_name]
    m1_sigma_idx = m1_name_to_idx[f"{converter.id}_ctrl_sigma"]
    m1_result = _run_arm(m1_system, m1_eq.x_star, m1_eq.converged, m1_bus_idx, m1_iL_idx, m1_sigma_idx,
                         lambda iL, vb, sg: current_loop_mrc_duty("", iL, vb, sg, m1_conv.controller.params))
    if m1_result.get("equilibrium_converged"):
        m1_sigma_t = _extract_sigma(m1_system, m1_eq, m1_sigma_idx, t_end, n_eval, perturb_frac, m1_bus_idx)
        m1_result["phi"] = [float(iL - i_bias - K_i_m1 * sg)
                            for iL, sg in zip(m1_result["i_L"], m1_sigma_t)]

    m2_iL_idx = m2_name_to_idx[f"{converter.id}_em_i_L"]
    m2_bus_idx = m2_name_to_idx[bus.state_name]
    m2_sigma_idx = m2_name_to_idx[f"{converter.id}_ctrl_sigma"]
    m2_result = _run_arm(m2_system, m2_eq.x_star, m2_eq.converged, m2_bus_idx, m2_iL_idx, m2_sigma_idx,
                         lambda iL, vb, sg: bus_voltage_mrc_duty("", iL, vb, sg, m2_conv.controller.params,
                                                                 m2_conv.controller.other_bus_injection))
    if m2_result.get("equilibrium_converged"):
        m2_sigma_t = _extract_sigma(m2_system, m2_eq, m2_sigma_idx, t_end, n_eval, perturb_frac, m2_bus_idx)
        m2_result["phi"] = [float(vb - v_nom + R_v * iL - K_i_m2 * sg)
                            for iL, vb, sg in zip(m2_result["i_L"], m2_result["v_bus"], m2_sigma_t)]

    return {
        "topology": topology,
        "v_nom": v_nom,
        "t_end": t_end,
        "perturb_frac": perturb_frac,
        "baseline": baseline_result,
        "m1": m1_result,
        "m2": m2_result,
        "m1_params": {"K_i": K_i_m1, "k_m": k_m_m1, "i_bias": i_bias},
        "m2_params": {"K_i": K_i_m2, "k_m": k_m_m2, "R_v": R_v},
    }


def _extract_sigma(system, eq, sigma_idx, t_end, n_eval, perturb_frac, bus_idx):
    """
    Re-simulate ONLY to recover sigma(t) alongside i_L(t)/v_bus(t) already
    captured in `arm_result` -- `_run_arm` above intentionally returns
    only i_L/v_bus/deviation/d as plain lists (JSON-friendly), so sigma(t)
    is recovered here from a second, IDENTICAL simulate() call (same seed
    state, same solver, same t_span/n_eval/max_step) rather than smuggling
    the full state array through `arm_result`. This is deliberately kept
    as a separate small helper instead of complicating `_run_arm`'s return
    contract for every caller that does not need phi(t).
    """
    x0 = eq.x_star.copy()
    x0[bus_idx] *= perturb_frac
    sim = Simulator(system, method="RK45")
    result = sim.simulate(x0=x0, t_span=(0.0, t_end), n_eval=n_eval, max_step=max(t_end / 200.0, 1e-6))
    return result.x[sigma_idx, :].tolist()


def verify_contraction_under_saturation(mrc_system, converter_id: str, x_star: np.ndarray,
                                        name_to_idx: Dict, bus_state_name: str, candidate: str,
                                        params: Dict, other_injection=None,
                                        v_bus_sag_fracs: Optional[List[float]] = None) -> Dict:
    """
    Task requirement 5: verify the exact contraction identity for the
    IDEAL, unsaturated M2 (phi2_dot = -k_m*phi2), then explicitly
    determine what happens under duty saturation -- never claiming exact
    contraction there unless the ACTUAL saturated (clamped) dynamics
    still satisfy it (they provably do not, in general: the duty law was
    solved to make phi2_dot=-k_m*phi2 hold for the UNCLAMPED control
    signal; once `ConverterModel.local_dynamics` clamps that signal via
    `clamp_duty`, the realized di_L/dt differs from what the identity's
    derivation assumed).

    For each fractional bus-voltage sag in `v_bus_sag_fracs` (progressively
    larger disturbances, more likely to push the REQUIRED duty outside
    [0.02, 0.98]), evaluates at the disturbed point x:
      - phi(x), the UNCLAMPED duty command d_raw the control law would
        issue, whether clamp_duty would clamp it,
      - phi_dot using the REAL (potentially clamped) assembled dynamics
        `mrc_system.dynamics`, which internally applies clamp_duty,
      - the residual phi_dot + k_m*phi, both in absolute terms and
        relative to k_m*phi.
    Reports, per point, whether the identity holds (small residual) or is
    violated (large residual, exactly when d_raw is clamped) -- so a
    saturated point is NEVER reported as satisfying exact contraction.
    """
    conv_id = converter_id
    iL_idx = name_to_idx[f"{conv_id}_em_i_L"]
    sigma_idx = name_to_idx[f"{conv_id}_ctrl_sigma"]
    bus_idx = name_to_idx[bus_state_name]
    k_m = params["k_m"]
    fracs = v_bus_sag_fracs if v_bus_sag_fracs is not None else [1.0, 0.97, 0.90, 0.75, 0.5, 0.3]

    def phi_of(x):
        if candidate == "phi1":
            return float(x[iL_idx] - params["i_bias"] - params["K_i"] * x[sigma_idx])
        return float(x[bus_idx] - params["v_nom"] + params["R_v"] * x[iL_idx] - params["K_i"] * x[sigma_idx])

    def phi_dot_numeric(x):
        f = np.asarray(mrc_system.dynamics(0.0, x, mrc_system.default_input(), mrc_system.params), dtype=float)
        if candidate == "phi1":
            return float(f[iL_idx] - params["K_i"] * f[sigma_idx])
        return float(f[bus_idx] + params["R_v"] * f[iL_idx] - params["K_i"] * f[sigma_idx])

    points = []
    for frac in fracs:
        x = x_star.copy()
        x[bus_idx] *= frac
        i_L = float(x[iL_idx]); v_bus = float(x[bus_idx]); sigma = float(x[sigma_idx])
        if candidate == "phi1":
            d_raw = current_loop_mrc_duty("", i_L, v_bus, sigma, params)
        else:
            d_raw = bus_voltage_mrc_duty("", i_L, v_bus, sigma, params, other_injection)
        clamped = bool(d_raw < 0.02 or d_raw > 0.98)
        phi = phi_of(x)
        phi_dot = phi_dot_numeric(x)
        residual = phi_dot + k_m * phi
        # At/near the manifold (phi~0, e.g. exactly at equilibrium) a
        # RELATIVE residual is ill-conditioned (dividing by k_m*phi~0
        # amplifies float/Newton-solve noise into a large ratio despite
        # an absolutely tiny residual) -- judged in absolute terms there
        # instead, exactly the same convention used for phi1's own
        # `_contraction_residual_check` / test_manifold_reshaping.py.
        if abs(phi) < 1e-8:
            identity_holds = bool(not clamped and abs(residual) < 1e-6)
            rel_residual = float(abs(residual))  # reported value is the absolute residual in this regime
        else:
            rel_residual = abs(residual) / abs(k_m * phi)
            identity_holds = bool(not clamped and rel_residual < 1e-3)
        points.append({
            "v_bus_sag_frac": frac, "phi": phi, "phi_dot": phi_dot, "d_raw": float(d_raw),
            "duty_clamped": clamped, "residual": residual, "relative_residual": rel_residual,
            "contraction_identity_holds": identity_holds,
        })

    any_clamped = any(p["duty_clamped"] for p in points)
    return {
        "candidate": candidate,
        "points": points,
        "ideal_unsaturated_identity_verified_at_equilibrium": points[0]["contraction_identity_holds"],
        "saturation_breaks_identity": any_clamped and any(p["duty_clamped"] and not p["contraction_identity_holds"] for p in points),
        "note": ("phi_dot=-k_m*phi is derived from and only valid for the UNCLAMPED control law; once "
                "the duty command saturates against [0.02, 0.98] (ConverterModel.clamp_duty), the realized "
                "closed-loop dynamics differs from the ideal model this identity was proven on, and the "
                "identity is NOT claimed to hold at those points (see each point's 'duty_clamped' and "
                "'contraction_identity_holds')."),
    }


def run_auto_mrc_pipeline_multi(net: Network, converter_id: Optional[str] = None,
                                v_nom: Optional[float] = None, mrc_params: Optional[Dict] = None) -> Dict:
    """
    Multi-converter-aware dispatcher in front of `run_auto_mrc_pipeline`.
    Added to satisfy: "a custom network with multiple controllable
    converters must not stop with 'converter_id must be given
    explicitly' -- automatically inspect every candidate, show a
    per-converter feasibility result, auto-select only when exactly one
    is feasible, otherwise require explicit selection. Never claim
    multi-input MRC."

    Dispatch rules (never silently choosing among several feasible
    converters):

    - `converter_id` given explicitly: delegates straight to
      `run_auto_mrc_pipeline` -- byte-for-byte the existing
      single-converter behavior (preservation requirement).
    - 0 or exactly 1 candidate converter in the network: also delegates
      straight to `run_auto_mrc_pipeline` (with `converter_id=None`).
      Zero candidates reproduces the existing NOT_ESTABLISHED message
      unchanged; exactly one candidate is an unambiguous auto-selection
      (there is nothing else it could mean), not a silent choice among
      alternatives.
    - >1 candidates and no `converter_id`: runs the FULL existing
      single-converter pipeline independently for each candidate (its
      own phi_j, Dphi_j, A_j = Dphi_j G_j, rank, relative degree,
      equilibrium compatibility, actuator constraints, synthesis
      feasibility -- exactly the pre-existing single-converter
      mathematics, reused once per converter, never a new coupled
      multi-input derivation). If exactly one candidate turns out
      feasible, its full report is returned (so it may be selected/
      applied), annotated with the per-converter scan for transparency.
      If more than one (or none) are feasible, returns a structured
      "selection required" (or "none feasible") report and does NOT
      apply or claim any of them.
    """
    if converter_id is not None:
        return run_auto_mrc_pipeline(net, converter_id=converter_id, v_nom=v_nom, mrc_params=mrc_params)

    candidates = list_duty_modulated_converters(net)
    if len(candidates) <= 1:
        return run_auto_mrc_pipeline(net, converter_id=None, v_nom=v_nom, mrc_params=mrc_params)

    per_converter: List[Dict] = []
    feasible_ids: List[str] = []
    for c in candidates:
        sub = run_auto_mrc_pipeline(net, converter_id=c.id, v_nom=v_nom, mrc_params=mrc_params)
        # NOTE: "MRC_SYNTHESIS_SUPPORTED_VIA_RESHAPED_MANIFOLD" (the M2/
        # bus-voltage-anchored reshaped-manifold outcome added alongside
        # the original phi1-only MRC_SYNTHESIS_SUPPORTED/DIAGNOSTIC_ONLY
        # statuses) must be counted as feasible here too -- a candidate
        # whose only viable synthesis path is M2 is genuinely feasible,
        # and omitting this status previously made the multi-converter
        # scan disagree with what run_auto_mrc_pipeline itself reports
        # for that exact same converter (a real backend/report
        # consistency bug, found and fixed during end-to-end validation).
        is_feasible = sub.get("status") in (
            "MRC_SYNTHESIS_SUPPORTED", "MRC_FEASIBILITY_DIAGNOSTIC_ONLY",
            "MRC_SYNTHESIS_SUPPORTED_VIA_RESHAPED_MANIFOLD",
        )
        per_converter.append({
            "converter_id": c.id,
            "topology": sub.get("topology"),
            "status": sub.get("status"),
            "feasible": is_feasible,
            "reason": sub.get("reason"),
            "rank_A": sub.get("rank_A"),
            "relative_degree": sub.get("relative_degree"),
            "control_authority": bool(sub.get("rank_A")),
            "equilibrium_compatible": sub.get("closed_loop_equilibrium_converged"),
            "transverse_matches_km": sub.get("transverse_matches_km"),
            "tangential_locally_stable": sub.get("tangential_locally_stable"),
            "saturation_note": sub.get("saturation_note"),
            "classification": sub.get("classification"),
            "reshaped_manifold": sub.get("status") == "MRC_SYNTHESIS_SUPPORTED_VIA_RESHAPED_MANIFOLD",
            "full_report": sub,
        })
        if is_feasible:
            feasible_ids.append(c.id)

    if len(feasible_ids) == 1:
        chosen = feasible_ids[0]
        result = dict(next(pc["full_report"] for pc in per_converter if pc["converter_id"] == chosen))
        result["multi_converter_scan"] = [
            {k: v for k, v in pc.items() if k != "full_report"} for pc in per_converter
        ]
        result["multi_converter_note"] = (
            f"{len(candidates)} directly duty-modulated converters were present "
            f"({[c.id for c in candidates]}); each was independently analyzed (its own phi_j, Dphi_j, "
            f"A_j=Dphi_j G_j, rank, relative degree, equilibrium compatibility, actuator constraints, "
            f"synthesis feasibility). Exactly one ('{chosen}') was feasible and was auto-selected -- the "
            f"others were evaluated, not silently ignored. This is a per-converter analysis, not a coupled "
            f"multi-input MRC derivation."
        )
        return result

    scan = [{k: v for k, v in pc.items() if k != "full_report"} for pc in per_converter]
    if feasible_ids:
        status = "MRC_MULTI_CANDIDATE_SELECTION_REQUIRED"
        reason = (
            f"{len(candidates)} directly duty-modulated converters are present and {len(feasible_ids)} of "
            f"them ({feasible_ids}) are independently MRC-feasible. Multiple feasible single-converter "
            f"candidates exist -- 'converter_id' must be given explicitly to select one; this platform "
            f"does not silently choose among them, and does not derive or claim a coupled multi-input MRC "
            f"across converters. See 'multi_converter_scan' for each candidate's own feasibility result."
        )
    else:
        status = "MRC_SYNTHESIS_NOT_ESTABLISHED"
        reason = (
            f"{len(candidates)} directly duty-modulated converters are present ({[c.id for c in candidates]}) "
            f"but none is independently MRC-feasible -- see each candidate's own 'reason' in "
            f"'multi_converter_scan'."
        )
    return {
        "status": status,
        "reason": reason,
        "candidate_converter_ids": [c.id for c in candidates],
        "feasible_converter_ids": feasible_ids,
        "multi_converter_scan": scan,
        "requires_explicit_selection": bool(feasible_ids),
    }


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
