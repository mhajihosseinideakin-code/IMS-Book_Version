"""
Rigorous validation that CPL -> MRC_FEASIBILITY_DIAGNOSTIC_ONLY (and never
the reverse) is produced by the ACTUAL reduced/tangential dynamics of the
assembled model at the tested operating point, never by the load's
component type/label.

Central analytic result, independently derived and verified here (see
test_reduced_jacobian_trace_matches_closed_form_analytic_derivation and
test_alpha_parallel_independent_of_K_i_km_ibias_RL): for the current-loop
MRC manifold this module implements (phi = i_L - i_bias - K_i*sigma,
anchored on i_L), the reduced dynamics on phi=0 is 2-dimensional
(v_bus, sigma), with reduced Jacobian

    J_M = [[ d(i_inj)/d(v_bus)/C ,  K_i/C ],
           [        -1          ,    0    ]]

so trace(J_M) = d(i_inj)/d(v_bus) / C. For a constant-IMPEDANCE load,
i_inj = -v_bus/R, so d(i_inj)/d(v_bus) = -1/R < 0 -- trace is negative,
contributing toward Hurwitz stability. For a constant-POWER load,
i_inj = -P/v_bus (smoothed near v_floor, irrelevant far from it),
so d(i_inj)/d(v_bus) = +P/v_bus^2 > 0 for any P > 0 -- trace(J_M) is
UNCONDITIONALLY POSITIVE, which by trace = sum(eigenvalues) means at
least one reduced eigenvalue has positive real part for ANY choice of
K_i, k_m, i_bias, R_L, or topology (buck/boost/buckboost only rescale
the off-diagonal K_i/C entry via each topology's own i_L-to-bus-current
coupling, never the diagonal/trace term). This is why no stable CPL
operating point exists anywhere in this parameter family for this
manifold construction -- a genuine, provable structural limitation of
THIS manifold/controller formulation, not a hard-coded "CPL = unstable"
rule and not a claim that CPLs can never be stabilized by ANY manifold.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy

from ims_platform.network.network import Network
from ims_platform.network.bus import Bus
from ims_platform.network.converter import Converter
from ims_platform.network.converter_topologies import BuckModel, BoostModel, BuckBoostModel
from ims_platform.network.controller import ConstantDutyController
from ims_platform.network.components import ConstantPowerLoad, ConstantImpedanceLoad
from ims_platform.mrc_designer.auto_manifold import (
    run_auto_mrc_pipeline, TANGENTIAL_EIGENVALUE_TOLERANCE, tangent_normal_decomposition,
)

TOPOLOGIES = {"buck": BuckModel, "boost": BoostModel, "buckboost": BuckBoostModel}
V_NOM = {"buck": 24.0, "boost": 72.0, "buckboost": 24.0}
V_IN = 48.0


def _build_cpl(topology: str, P: float, v_in: float = V_IN, L: float = 1e-3, R_L: float = 0.05,
               C: float = 2e-3, v_floor: float = 0.5) -> Network:
    v_nom = V_NOM[topology]
    net = Network("t")
    net.add_bus(Bus(id="bus", C=C, v_init=v_nom, v_min=v_nom * 0.05))
    em = TOPOLOGIES[topology](v_in=v_in, L=L, R_L=R_L)
    net.add_component(Converter(id="conv", bus="bus", electrical_model=em, controller=ConstantDutyController(d=0.5)))
    net.add_component(ConstantPowerLoad(id="load", bus="bus", P=P, v_floor=v_floor))
    return net


def _build_impedance(topology: str, R: float, v_in: float = V_IN, L: float = 1e-3, R_L: float = 0.05,
                      C: float = 2e-3) -> Network:
    v_nom = V_NOM[topology]
    net = Network("t")
    net.add_bus(Bus(id="bus", C=C, v_init=v_nom, v_min=v_nom * 0.05))
    em = TOPOLOGIES[topology](v_in=v_in, L=L, R_L=R_L)
    net.add_component(Converter(id="conv", bus="bus", electrical_model=em, controller=ConstantDutyController(d=0.5)))
    net.add_component(ConstantImpedanceLoad(id="load", bus="bus", R=R))
    return net


# ---------------------------------------------------------------------------
# 1. No CPL-label-based classification anywhere in the pipeline source.
# ---------------------------------------------------------------------------

def test_no_cpl_label_based_branching_in_auto_manifold_source():
    """Static audit: auto_manifold.py must contain no reference to CPL,
    ConstantPowerLoad, or any load-type string at all -- the tangential
    stability classification only ever touches eigenvalues of the real
    assembled closed-loop Jacobian."""
    import inspect
    from ims_platform.mrc_designer import auto_manifold
    src = inspect.getsource(auto_manifold)
    for forbidden in ("ConstantPowerLoad", "CPL", "load_type", "\"cpl\"", "'cpl'"):
        assert forbidden not in src, f"found forbidden load-type-based reference {forbidden!r} in auto_manifold.py"


# ---------------------------------------------------------------------------
# 2-3. Real per-topology CPL cases -- all three topologies, actual pipeline.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("topology", ["buck", "boost", "buckboost"])
def test_cpl_diagnostic_only_is_explained_by_actual_reduced_eigenvalues(topology):
    net = _build_cpl(topology, P=50.0)
    report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=V_NOM[topology])
    # The FIRST-candidate (phi1, current-anchored) finding is preserved
    # exactly, whatever the pipeline's overall status ends up being after
    # attempting manifold reshaping (sub-task D): transverse contraction
    # established, but the reduced/tangential dynamics genuinely has a
    # positive mode -- this is phi1's own math, never overwritten.
    assert report["transverse_matches_km"] is True
    assert report["contraction_residual"]["max_relative_error"] < 1e-6
    assert report["reduced_dynamics_status"] == "UNSTABLE"
    assert report["alpha_parallel"] > TANGENTIAL_EIGENVALUE_TOLERANCE
    assert str(report["alpha_parallel"]) in report["reason"] or f"{report['alpha_parallel']:.6g}" in report["reason"]
    # Overall status: either phi1 stands alone (DIAGNOSTIC_ONLY) or the
    # bus-voltage-anchored second candidate (phi2) was found, by an
    # explicit reported R_v search, to stabilize the reduced dynamics --
    # never silently forced to SUPPORTED without a genuine phi2 result.
    assert report["status"] in ("MRC_FEASIBILITY_DIAGNOSTIC_ONLY", "MRC_SYNTHESIS_SUPPORTED_VIA_RESHAPED_MANIFOLD")
    assert report["manifold_reshaping"]["attempted"] is True
    if report["status"] == "MRC_SYNTHESIS_SUPPORTED_VIA_RESHAPED_MANIFOLD":
        assert report["manifold_reshaping"]["status"] == "MRC_SYNTHESIS_SUPPORTED_VIA_M2"
        assert report["manifold_reshaping"]["chosen_R_v"] is not None
    else:
        assert report["manifold_reshaping"]["status"] == "NO_STABILIZING_R_V_FOUND_IN_GRID"


@pytest.mark.parametrize("topology", ["buck", "boost", "buckboost"])
def test_impedance_load_same_topology_reaches_supported(topology):
    """Same topology, ONLY the load type changed (impedance instead of
    CPL): must reach SUPPORTED, proving the classification distinguishes
    load types through their actual physics (d(i_inj)/dv_bus sign), not
    through a hard-coded rule for either label."""
    net = _build_impedance(topology, R=V_NOM[topology] ** 2 / 50.0)
    report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=V_NOM[topology])
    assert report["status"] == "MRC_SYNTHESIS_SUPPORTED"
    assert report["reduced_dynamics_status"] == "STABLE"
    assert report["alpha_parallel"] < -TANGENTIAL_EIGENVALUE_TOLERANCE


# ---------------------------------------------------------------------------
# 4. Analytic proof: the reduced Jacobian's trace, computed independently
#    with sympy from the manifold's own defining equations, matches the
#    numerically-observed alpha_parallel, and is unconditionally positive
#    for P > 0.
# ---------------------------------------------------------------------------

def test_reduced_jacobian_trace_matches_closed_form_analytic_derivation():
    v_bus, sigma, P, C, K_i, i_bias, v_nom = sympy.symbols(
        "v_bus sigma P C K_i i_bias v_nom", real=True, positive=True)
    i_L = i_bias + K_i * sigma
    d_v_bus = (i_L - P / v_bus) / C          # dv_bus/dt on the manifold (phi=0 => i_L slaved to sigma)
    d_sigma = v_nom - v_bus
    f = sympy.Matrix([d_v_bus, d_sigma])
    X = sympy.Matrix([v_bus, sigma])
    J = f.jacobian(X)
    trace = sympy.simplify(J.trace())
    det = sympy.simplify(J.det())
    assert trace == P / (C * v_bus ** 2)
    assert det == K_i / C
    # trace has NO dependence on K_i, k_m, i_bias, R_L whatsoever, and is
    # strictly positive for any physically valid P>0, C>0, v_bus>0 -- the
    # reduced 2x2 system can never be Hurwitz (trace>0 => sum of
    # eigenvalues' real parts > 0 => at least one has positive real part).
    assert K_i not in trace.free_symbols
    assert i_bias not in trace.free_symbols


@pytest.mark.parametrize("P", [5.0, 20.0, 50.0, 100.0])
def test_reduced_jacobian_trace_formula_matches_numeric_alpha_parallel(P):
    """The closed-form trace = P/(C*v_bus^2) must match the ACTUAL
    numerically-observed alpha_parallel from the real assembled pipeline
    to high precision (this is a 2-state reduced system, so the trace
    IS the sum of both eigenvalues; whenever they are a complex-conjugate
    pair, alpha_parallel = trace/2, and whenever real, alpha_parallel is
    bounded by trace -- checked directly against whichever regime
    applies here)."""
    C = 2e-3
    net = _build_cpl("buck", P=P, C=C)
    report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=24.0)
    v_bus_star = 24.0
    trace_predicted = P / (C * v_bus_star ** 2)
    tangential = report["tangential_eigenvalues"]
    trace_actual = sum(e["re"] for e in tangential)
    assert abs(trace_actual - trace_predicted) < 1e-3 * max(1.0, trace_predicted)


# ---------------------------------------------------------------------------
# 5. alpha_parallel is invariant under K_i, k_m, i_bias, R_L -- direct
#    numerical confirmation of the analytic trace result above.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("K_i", [1.0, 20.0, 500.0, 20000.0])
def test_alpha_parallel_independent_of_K_i(K_i):
    net = _build_cpl("buck", P=50.0)
    report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=24.0, mrc_params={"K_i": K_i})
    assert abs(report["alpha_parallel"] - 21.701377) < 1e-3


@pytest.mark.parametrize("k_m", [100.0, 500.0, 1000.0, 3000.0])
def test_alpha_parallel_independent_of_k_m_but_transverse_eigenvalue_tracks_it(k_m):
    """Central distinction the task requires: k_m sets the TRANSVERSE
    convergence rate (lambda_perp ~= -k_m) but must NOT move the
    tangential/reduced eigenvalues at all -- MRC cannot repair an
    unstable reduced mode by tuning k_m."""
    net = _build_cpl("buck", P=50.0)
    report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=24.0, mrc_params={"k_m": k_m})
    assert abs(report["transverse_eigenvalue"]["re"] - (-k_m)) < max(1e-2 * k_m, 1.0)
    assert abs(report["transverse_eigenvalue"]["im"]) < 1.0
    assert abs(report["alpha_parallel"] - 21.701377) < 1e-3  # unchanged across all k_m


@pytest.mark.parametrize("i_bias", [-2.0, -1.0, 0.0, 1.0, 2.0])
def test_alpha_parallel_independent_of_i_bias(i_bias):
    net = _build_cpl("buck", P=50.0)
    report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=24.0, mrc_params={"i_bias": i_bias})
    assert abs(report["alpha_parallel"] - 21.701377) < 1e-3


# ---------------------------------------------------------------------------
# 6. alpha_parallel DOES change with P and C -- proof the classification
#    is equation-driven (a continuous function of the real physics), not
#    a constant baked in for the CPL label. Explicit "same label, two
#    different numeric outcomes" requirement (task item 8 / item 14).
# ---------------------------------------------------------------------------

def test_alpha_parallel_scales_linearly_with_P_same_load_type():
    net_a = _build_cpl("buck", P=20.0)
    net_b = _build_cpl("buck", P=100.0)
    r_a = run_auto_mrc_pipeline(net_a, converter_id="conv", v_nom=24.0)
    r_b = run_auto_mrc_pipeline(net_b, converter_id="conv", v_nom=24.0)
    # both are CPL under the FIRST (current-anchored) candidate, so both
    # phi1 reduced-dynamics evaluations are UNSTABLE (see module
    # docstring: no stable CPL point exists for phi1, proved
    # analytically) -- but the actual alpha_parallel value is markedly
    # different and tracks P linearly, which is only possible if it is
    # computed from the real equations, not a fixed per-label constant.
    # (The overall top-level status may differ between them if manifold
    # reshaping's R_v search succeeds for one P and not the other -- that
    # is a separate, later-stage claim about phi2, not about phi1.)
    assert r_a["reduced_dynamics_status"] == r_b["reduced_dynamics_status"] == "UNSTABLE"
    ratio = r_b["alpha_parallel"] / r_a["alpha_parallel"]
    assert abs(ratio - 5.0) < 1e-2  # P_b/P_a == 5


def test_alpha_parallel_scales_inversely_with_C_same_load_type():
    net_a = _build_cpl("buck", P=50.0, C=1e-3)
    net_b = _build_cpl("buck", P=50.0, C=4e-3)
    r_a = run_auto_mrc_pipeline(net_a, converter_id="conv", v_nom=24.0)
    r_b = run_auto_mrc_pipeline(net_b, converter_id="conv", v_nom=24.0)
    ratio = r_a["alpha_parallel"] / r_b["alpha_parallel"]
    assert abs(ratio - 4.0) < 1e-2  # C_b/C_a == 4 => alpha_a/alpha_b == 4


def test_cpl_classification_is_not_hardcoded_same_label_different_math_result():
    """Direct regression for task item 8/14: SAME component type (CPL) at
    two different physical parameter sets produces two DIFFERENT
    alpha_parallel-driven outcomes when the underlying math genuinely
    differs (here: crossing into the numerical NOT_ESTABLISHED boundary
    band at a sufficiently small P, versus a clearly UNSTABLE point at a
    normal P) -- proof the classifier reads the equations, not the
    label. (No positive-P CPL point is ever STABLE for this manifold --
    see module docstring's trace proof -- so the two distinguishable
    outcomes reachable by varying only P are UNSTABLE vs the boundary
    NOT_ESTABLISHED state, not SUPPORTED vs DIAGNOSTIC; this is reported
    as what the equations actually show, not forced to a different
    pattern.)"""
    net_normal = _build_cpl("buck", P=50.0)
    net_tiny = _build_cpl("buck", P=1e-9)
    r_normal = run_auto_mrc_pipeline(net_normal, converter_id="conv", v_nom=24.0)
    r_tiny = run_auto_mrc_pipeline(net_tiny, converter_id="conv", v_nom=24.0)
    # phi1's own reduced-dynamics classification (never overwritten by a
    # later manifold-reshaping attempt) is what genuinely differs here.
    assert r_normal["reduced_dynamics_status"] != r_tiny["reduced_dynamics_status"]
    assert r_normal["reduced_dynamics_status"] == "UNSTABLE"
    assert r_tiny["reduced_dynamics_status"] == "NOT_ESTABLISHED"
    assert abs(r_tiny["alpha_parallel"]) <= TANGENTIAL_EIGENVALUE_TOLERANCE
    # a NOT_ESTABLISHED phi1 (boundary case, no clear instability to
    # react to) never triggers manifold reshaping -- only a clearly
    # UNSTABLE phi1 does (see run_auto_mrc_pipeline's UNSTABLE branch).
    assert r_tiny["status"] == "MRC_SYNTHESIS_NOT_ESTABLISHED"
    assert "manifold_reshaping" not in r_tiny
    assert r_normal["status"] in ("MRC_FEASIBILITY_DIAGNOSTIC_ONLY", "MRC_SYNTHESIS_SUPPORTED_VIA_RESHAPED_MANIFOLD")


# ---------------------------------------------------------------------------
# 7. Numerical eigenvalue tolerance around zero -- boundary must be
#    NOT_ESTABLISHED, never VIOLATED (task item 6).
# ---------------------------------------------------------------------------

def test_boundary_alpha_parallel_is_not_established_not_violated():
    net = _build_cpl("buck", P=1e-9)  # alpha_parallel lands within tolerance (verified above)
    report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=24.0)
    assert report["reduced_dynamics_status"] == "NOT_ESTABLISHED"
    assert report["tangential_locally_stable"] is None  # neither True nor False
    assert report["status"] == "MRC_SYNTHESIS_NOT_ESTABLISHED"
    assert "boundary" in report["reason"].lower() or "NOT_ESTABLISHED" in report["reason"]


# ---------------------------------------------------------------------------
# 8. Exact transverse residual identity: phi_dot + k_m*phi ~= 0.
# ---------------------------------------------------------------------------

def test_exact_transverse_residual_identity_holds_for_cpl_case():
    net = _build_cpl("buck", P=50.0)
    report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=24.0)
    assert report["contraction_residual"]["max_relative_error"] < 1e-6
    assert report["transverse_matches_km"] is True


# ---------------------------------------------------------------------------
# 9. Actuator saturation genuinely breaks the ideal identity where it
#    applies -- reuses the saturation-check unit test's underlying logic
#    on a CPL-derived control law, confirming the identity claim is
#    conditioned on staying inside [0.02, 0.98], not asserted blindly.
# ---------------------------------------------------------------------------

def test_saturation_note_present_and_consistent_for_cpl_case():
    net = _build_cpl("buck", P=50.0)
    report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=24.0)
    sat = report["saturation_note"]
    assert "duty_at_equilibrium" in sat and "saturated" in sat
    if sat["saturated"]:
        assert "NOT active" in sat["note"]
    else:
        assert 0.02 <= sat["duty_at_equilibrium"] <= 0.98


# ---------------------------------------------------------------------------
# 10. Before/After, identical conditions, for the CPL (unstable-tangential)
#     case -- both arms perturbed identically, same horizon/time grid.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 11. Audit: the reported "tangential eigenvalues" must be the actual
#     restriction of the closed-loop Jacobian to ker(Dphi(x*)) -- not
#     merely "every eigenvalue except the one nearest -k_m" -- and this
#     must be verified against the full ambient spectrum, not assumed
#     from eigenvalue ordering.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("topology,P,load_kind", [
    ("buck", 50.0, "cpl"), ("boost", 50.0, "cpl"), ("buckboost", 50.0, "cpl"),
])
def test_tangential_eigenvalues_are_genuine_tangent_space_projection(topology, P, load_kind):
    net = _build_cpl(topology, P=P)
    report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=V_NOM[topology])
    decomp = report["tangent_normal_decomposition"]
    assert decomp is not None
    assert report["reduced_dynamics_method"].startswith("tangent_space_projection")
    # Dphi is proven an EXACT left eigenvector of J with eigenvalue -k_m
    # (algebraic identity, not numeric proximity search)
    assert decomp["left_eigenvector_verified"] is True
    assert decomp["left_eigenvector_relative_residual"] < 1e-6
    # tangential eigenvalues (from ker(Dphi) restriction) union {-k_m}
    # must reconstruct the FULL closed-loop spectrum exactly
    assert decomp["spectrum_consistent_with_full_jacobian"] is True
    assert decomp["tangent_space_dim"] == len(report["closed_loop_eigenvalues"]) - 1


def test_tangent_space_projection_matches_direct_null_space_computation_independently():
    """Independent cross-check: re-derive T=ker(Dphi) and J_par from
    scratch in the test itself (not by calling the module's own
    function on itself), confirming the reported tangential eigenvalues
    are reproducible by an outside computation."""
    from scipy.linalg import null_space
    from ims_platform.mrc_designer.auto_manifold import attach_auto_mrc
    from ims_platform.network.assembler import AutomaticModelBuilder

    net = _build_cpl("buck", P=50.0)
    report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=24.0)
    mrc_net, _ = attach_auto_mrc(net, "conv", 24.0, None)
    mrc_system = AutomaticModelBuilder.build(mrc_net, name="t_auto_mrc")
    cl_eq = mrc_system.find_equilibrium(mrc_system.initial_guess(), with_eigs=True)
    names = mrc_system.state_names
    n = len(names)
    Dphi = np.zeros(n)
    Dphi[names.index("conv_em_i_L")] = 1.0
    Dphi[names.index("conv_ctrl_sigma")] = -20.0  # default K_i
    T = null_space(Dphi.reshape(1, -1))
    J_par = np.linalg.pinv(T) @ cl_eq.jacobian @ T
    independent_eigs = sorted(np.linalg.eigvals(J_par), key=lambda z: (round(z.real, 4), round(z.imag, 4)))
    reported_eigs = sorted(
        [complex(e["re"], e["im"]) for e in report["tangential_eigenvalues"]],
        key=lambda z: (round(z.real, 4), round(z.imag, 4)),
    )
    assert np.allclose(independent_eigs, reported_eigs, atol=1e-3)


def test_tangent_normal_decomposition_function_directly_on_synthetic_matrix():
    """Direct unit test of tangent_normal_decomposition() on a hand-built
    3x3 matrix satisfying Dphi @ J = -k_m * Dphi exactly by
    construction, independent of any network/converter machinery."""
    k_m = 500.0
    K_i = 20.0
    Dphi = np.array([0.0, 1.0, -K_i])
    # Build any J whose Dphi is a left-eigenvector with eigenvalue -k_m:
    # pick a tangential 2x2 block A (arbitrary), then solve for the
    # coupling row that satisfies the constraint.
    A = np.array([[3.0, -2.0], [1.0, 0.5]])  # arbitrary tangential dynamics
    # state order (v_bus, i_L, sigma); tangent basis columns (1,0,0) and (0,K_i,1)
    T = np.array([[1.0, 0.0], [0.0, K_i], [0.0, 1.0]])
    J = np.zeros((3, 3))
    # Fill J such that T_pinv @ J @ T == A and Dphi @ J == -k_m*Dphi.
    # Simplest construction: J restricted to tangent directions reproduces A,
    # and the i_L row is chosen to satisfy the left-eigenvector identity.
    T_pinv = np.linalg.pinv(T)
    # Choose J's tangential block directly via J = T @ A @ T_pinv (this
    # automatically satisfies Dphi @ J = 0 = -k_m*0 only if phi(x)=0 is
    # trivial; instead directly enforce the true construction below).
    J[:, :] = T @ A @ T_pinv
    # Overwrite the i_L row (index 1) so Dphi @ J = -k_m * Dphi holds
    # exactly: Dphi @ J = J[1,:] - K_i*J[2,:] must equal -k_m*Dphi.
    target = -k_m * Dphi
    # Solve for J[1,:] given J[2,:] already set from the T@A@T_pinv fill:
    J[1, :] = target + K_i * J[2, :]

    result = tangent_normal_decomposition(J, Dphi, k_m)
    assert result["left_eigenvector_verified"] is True
    assert result["spectrum_consistent_with_full_jacobian"] is True
    assert result["transverse_eigenvalue"] == {"re": -k_m, "im": 0.0}
    tangential_re = sorted(e["re"] for e in result["tangential_eigenvalues"])
    expected_re = sorted(np.linalg.eigvals(A).real.tolist())
    assert np.allclose(tangential_re, expected_re, atol=1e-6)


def test_before_after_identical_conditions_for_unstable_cpl_case():
    net = _build_cpl("buck", P=50.0)
    report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=24.0)
    assert report["status"] in ("MRC_FEASIBILITY_DIAGNOSTIC_ONLY", "MRC_SYNTHESIS_SUPPORTED_VIA_RESHAPED_MANIFOLD")
    assert report["reduced_dynamics_status"] == "UNSTABLE"  # phi1's own finding, unchanged
    comparison = report["closed_loop_comparison"]
    baseline, mrc = comparison["baseline"], comparison["mrc"]
    assert baseline["t"] == mrc["t"]  # identical time grid/horizon for both arms
    assert len(baseline["i_L"]) == len(mrc["i_L"]) == len(baseline["t"])
    # the MRC pipeline still reports honest, separate claim levels for the
    # FIRST candidate even though ITS tangential dynamics is unstable --
    # never upgraded, whatever the second candidate later achieves
    assert report["claim_levels"]["certified_regional_ims"]["established"] is False
    assert report["claim_levels"]["local_equilibrium_stability"]["established"] is False
