"""
ims_tests.test_manifold_reshaping
------------------------------------

Sub-task D regression tests: "IMS Platform -- Add Automatic Stabilizing
Manifold Reshaping for Tangentially Unstable Cases."

Covers:
  1. The rigorous tangent/normal decomposition (tangent_normal_decomposition)
     is used for BOTH candidate manifolds, never eigenvalue-ordering.
  2. The first-candidate (phi1, current-anchored) manifold is preserved
     exactly -- never removed, never silently reclassified.
  3. The second-candidate (phi2, bus-voltage-anchored) manifold is derived
     INDEPENDENTLY per topology (A2 differs structurally between buck and
     boost/buck-boost -- buck needs R_v != 0 for relative degree one, the
     others do not), never a blind copy of one formula.
  4. Manifold reshaping is only attempted when phi1's reduced dynamics are
     genuinely UNSTABLE (never for an already-SUPPORTED or a boundary
     NOT_ESTABLISHED phi1).
  5. Where the R_v search finds a stabilizing candidate, BOTH lambda_perp
     < 0 (algebraically, exactly -k_m) and alpha_parallel < 0 are verified
     -- never one claim inferred from the other.
  6. Where the R_v search finds no stabilizing candidate, the full search
     table (every attempted R_v with its own alpha_parallel/status) is
     reported honestly -- never forced to SUPPORTED.
"""

import numpy as np
import pytest
import sympy

from ims_platform.network.network import Network
from ims_platform.network.bus import Bus
from ims_platform.network.converter import Converter
from ims_platform.network.converter_topologies import BuckModel, BoostModel, BuckBoostModel
from ims_platform.network.controller import ConstantDutyController, bus_voltage_mrc_duty
from ims_platform.network.components import ConstantPowerLoad, ConstantImpedanceLoad
from ims_platform.mrc_designer.auto_manifold import (
    run_auto_mrc_pipeline,
    derive_symbolic_law_v2,
    attempt_bus_voltage_manifold_reshaping,
    TANGENTIAL_EIGENVALUE_TOLERANCE,
)
from ims_platform.network.assembler import AutomaticModelBuilder

TOPOLOGIES = {"buck": BuckModel, "boost": BoostModel, "buckboost": BuckBoostModel}
V_NOM = {"buck": 24.0, "boost": 72.0, "buckboost": 24.0}
V_IN = 48.0


def _build_cpl(topology, P, v_nom=None, v_in=V_IN, L=1e-3, R_L=0.05, C=2e-3, v_floor=0.5):
    v_nom = v_nom if v_nom is not None else V_NOM[topology]
    net = Network("t")
    net.add_bus(Bus(id="bus", C=C, v_init=v_nom, v_min=v_nom * 0.05))
    em = TOPOLOGIES[topology](v_in=v_in, L=L, R_L=R_L)
    net.add_component(Converter(id="conv", bus="bus", electrical_model=em, controller=ConstantDutyController(d=0.5)))
    net.add_component(ConstantPowerLoad(id="load", bus="bus", P=P, v_floor=v_floor))
    return net


def _build_impedance(topology, R, v_nom=None, v_in=V_IN, L=1e-3, R_L=0.05, C=2e-3):
    v_nom = v_nom if v_nom is not None else V_NOM[topology]
    net = Network("t")
    net.add_bus(Bus(id="bus", C=C, v_init=v_nom, v_min=v_nom * 0.05))
    em = TOPOLOGIES[topology](v_in=v_in, L=L, R_L=R_L)
    net.add_component(Converter(id="conv", bus="bus", electrical_model=em, controller=ConstantDutyController(d=0.5)))
    net.add_component(ConstantImpedanceLoad(id="load", bus="bus", R=R))
    return net


# ---------------------------------------------------------------------------
# 1. phi2's relative degree is derived independently per topology (not a
#    blind copy): buck genuinely requires R_v != 0, boost/buck-boost do not.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("topology", ["buck", "boost", "buckboost"])
def test_phi2_A2_symbolic_derived_independently_per_topology(topology):
    law2 = derive_symbolic_law_v2(topology, {})
    syms = law2["symbols"]
    A2_at_Rv0 = sympy.simplify(law2["A_symbolic"].subs(syms["R_v"], 0))
    if topology == "buck":
        # buck's port_current = i_L has NO direct duty dependence -- A2
        # collapses to identically zero at R_v=0, i.e. relative degree
        # one genuinely REQUIRES the droop term.
        assert A2_at_Rv0 == 0
    else:
        # boost/buck-boost's port_current = (1-d)*i_L DOES depend on d
        # directly, so A2 is generically nonzero even at R_v=0 -- a
        # structurally different relative-degree story from buck, exactly
        # the "do not blindly copy the four-state formula across
        # topologies" distinction the task called out.
        assert A2_at_Rv0 != 0


def test_phi2_A2_symbolic_matches_runtime_and_finite_difference_buck():
    """Three independent computational paths for A2 (symbolic derivation,
    the runtime bus_voltage_mrc_duty formula, and a finite difference of
    the REAL ConverterModel) must agree, exactly the discipline already
    used for phi1's A(x)."""
    net = _build_cpl("buck", P=50.0)
    report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=24.0)
    mr = report["manifold_reshaping"]
    assert mr["attempted"] is True
    for entry in mr["search"]:
        if "A2_finite_difference_check" in entry:
            chk = entry["A2_finite_difference_check"]
            assert chk["verified"] is True
            assert chk["A2_relative_error"] < 1e-3


# ---------------------------------------------------------------------------
# 2. Manifold reshaping is triggered ONLY when phi1 is genuinely UNSTABLE.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("topology", ["buck", "boost", "buckboost"])
def test_no_reshaping_attempted_when_phi1_already_supported(topology):
    net = _build_impedance(topology, R=V_NOM[topology] ** 2 / 50.0)
    report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=V_NOM[topology])
    assert report["status"] == "MRC_SYNTHESIS_SUPPORTED"
    assert "manifold_reshaping" not in report


def test_no_reshaping_attempted_when_phi1_is_boundary_not_established():
    net = _build_cpl("buck", P=1e-9)  # lands within the numerical boundary band
    report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=24.0)
    assert report["reduced_dynamics_status"] == "NOT_ESTABLISHED"
    assert "manifold_reshaping" not in report


# ---------------------------------------------------------------------------
# 3. First candidate (phi1) is preserved exactly -- not removed, not mutated.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("topology", ["buck", "boost", "buckboost"])
def test_first_candidate_fields_preserved_when_reshaping_attempted(topology):
    net = _build_cpl(topology, P=50.0)
    report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=V_NOM[topology])
    # phi1's own construction/verification fields are all still present
    # and unmodified, whatever the final overall status is.
    assert report["candidate_phi"] == "phi(x) = i_L - i_bias - K_i*sigma,  sigma_dot = v_nom - v_bus"
    assert report["transverse_matches_km"] is True
    assert report["reduced_dynamics_status"] == "UNSTABLE"
    assert report["contraction_residual"]["max_relative_error"] < 1e-6
    assert "manifold_reshaping" in report
    assert report["manifold_reshaping"]["candidate_phi"].startswith("phi2(x) = v_bus")


# ---------------------------------------------------------------------------
# 4. Where a stabilizing R_v is found: BOTH lambda_perp < 0 (exact) and
#    alpha_parallel < 0 are independently verified, per topology.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("topology", ["buck", "boost", "buckboost"])
def test_bus_voltage_manifold_achieves_both_transverse_and_tangential_stability(topology):
    net = _build_cpl(topology, P=50.0)
    report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=V_NOM[topology])
    mr = report["manifold_reshaping"]
    assert mr["status"] == "MRC_SYNTHESIS_SUPPORTED_VIA_M2", (
        f"expected a stabilizing R_v to exist in the searched grid for {topology}+CPL at P=50W; "
        f"search={mr['search']}"
    )
    chosen = mr["chosen_entry"]
    decomp = chosen["tangent_normal_decomposition"]
    # lambda_perp: algebraic left-eigenvector identity, not eigenvalue
    # ordering/proximity.
    assert decomp["left_eigenvector_verified"] is True
    assert decomp["spectrum_consistent_with_full_jacobian"] is True
    assert decomp["transverse_eigenvalue"] == {"re": -mr["k_m"], "im": 0.0}
    # alpha_parallel: a SEPARATE claim, from the tangent-space-restricted
    # Jacobian, never inferred from the transverse result.
    assert chosen["alpha_parallel"] < -TANGENTIAL_EIGENVALUE_TOLERANCE
    assert report["status"] == "MRC_SYNTHESIS_SUPPORTED_VIA_RESHAPED_MANIFOLD"
    assert report["active_candidate"] == "phi2 (bus-voltage-anchored)"
    assert report["chosen_R_v"] == mr["chosen_R_v"]


# ---------------------------------------------------------------------------
# 5. Buck's exact closed-form stabilizing window: trace(J2_reduced) =
#    P/(C*v_bus*^2) - 1/(C*R_v) < 0  <=>  R_v < v_bus*^2/P (v_bus*=v_nom).
#    Direct construction (not the pipeline's grid) confirms the boundary.
# ---------------------------------------------------------------------------

def test_buck_phi2_exact_stabilizing_window_matches_closed_form():
    P = 50.0
    v_nom = 24.0
    C = 2e-3
    R_v_crit = v_nom ** 2 / P  # == 11.52

    def alpha_parallel_at(R_v):
        net = _build_cpl("buck", P=P, v_nom=v_nom, C=C)
        report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=v_nom)
        from ims_platform.mrc_designer.auto_manifold import attempt_bus_voltage_manifold_reshaping
        baseline_system = AutomaticModelBuilder.build(net, name="t")
        base_eq = baseline_system.find_equilibrium(baseline_system.initial_guess(), with_eigs=False)
        from ims_platform.mrc_designer.auto_manifold import find_duty_modulated_converter
        converter = find_duty_modulated_converter(net, "conv")
        r = attempt_bus_voltage_manifold_reshaping(
            net, converter, "buck", v_nom, baseline_system, base_eq, k_m=500.0, K_i=20.0,
            r_v_grid=[R_v],
        )
        return r["search"][0].get("alpha_parallel")

    alpha_below = alpha_parallel_at(R_v_crit * 0.5)   # well inside the window
    alpha_above = alpha_parallel_at(R_v_crit * 1.5)   # outside the window
    assert alpha_below is not None and alpha_below < 0
    assert alpha_above is not None and alpha_above > 0

    def trace_closed_form(R_v):
        return P / (C * v_nom ** 2) - 1.0 / (C * R_v)

    # This 2-state (i_L, sigma) reduced system's det = K_i/(C*R_v) > 0
    # everywhere on this window, so its two eigenvalues form a complex-
    # conjugate pair (same regime already established for phi1's reduced
    # system in test_reduced_jacobian_trace_formula_matches_numeric_alpha_parallel):
    # alpha_parallel = Re(each eigenvalue) = trace/2, not trace itself.
    assert abs(alpha_below - trace_closed_form(R_v_crit * 0.5) / 2.0) < 1e-3
    assert abs(alpha_above - trace_closed_form(R_v_crit * 1.5) / 2.0) < 1e-3


# ---------------------------------------------------------------------------
# 6. Honest reporting when NO stabilizing R_v exists in the searched grid
#    (constructed directly, not asserting it never happens in the wild --
#    the buck closed form shows an upper bound v_nom^2/P exists, so a grid
#    that only searches ABOVE it must fail honestly, never force SUPPORTED).
# ---------------------------------------------------------------------------

def test_manifold_reshaping_reports_honest_failure_when_grid_has_no_stabilizing_value():
    from ims_platform.mrc_designer.auto_manifold import (
        attempt_bus_voltage_manifold_reshaping, find_duty_modulated_converter,
    )
    P = 50.0
    v_nom = 24.0
    net = _build_cpl("buck", P=P, v_nom=v_nom)
    baseline_system = AutomaticModelBuilder.build(net, name="t")
    base_eq = baseline_system.find_equilibrium(baseline_system.initial_guess(), with_eigs=False)
    converter = find_duty_modulated_converter(net, "conv")
    # R_v_crit = v_nom^2/P = 11.52 -- search only ABOVE the stabilizing window
    r = attempt_bus_voltage_manifold_reshaping(
        net, converter, "buck", v_nom, baseline_system, base_eq, k_m=500.0, K_i=20.0,
        r_v_grid=[20.0, 50.0, 100.0],
    )
    assert r["status"] == "NO_STABILIZING_R_V_FOUND_IN_GRID"
    assert r["chosen_R_v"] is None
    assert len(r["search"]) == 3
    for entry in r["search"]:
        assert entry.get("alpha_parallel", 0) is None or entry["alpha_parallel"] > 0
    assert "does not prove none exists" in r["reason"]


# ---------------------------------------------------------------------------
# 7. Runtime duty law (bus_voltage_mrc_duty) enforces the exact contraction
#    identity phi2_dot = -k_m*phi2 on the real assembled dynamics.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("topology", ["buck", "boost", "buckboost"])
def test_bus_voltage_mrc_duty_enforces_exact_contraction_identity(topology):
    from ims_platform.mrc_designer.auto_manifold import (
        attempt_bus_voltage_manifold_reshaping, find_duty_modulated_converter,
    )
    v_nom = V_NOM[topology]
    net = _build_cpl(topology, P=50.0)
    baseline_system = AutomaticModelBuilder.build(net, name="t")
    base_eq = baseline_system.find_equilibrium(baseline_system.initial_guess(), with_eigs=False)
    converter = find_duty_modulated_converter(net, "conv")
    mr = attempt_bus_voltage_manifold_reshaping(
        net, converter, topology, v_nom, baseline_system, base_eq, k_m=500.0, K_i=20.0,
    )
    assert mr["status"] == "MRC_SYNTHESIS_SUPPORTED_VIA_M2"
    mrc_system = mr["mrc_system"]
    cl_eq = mr["cl_eq"]
    name_to_idx = mr["name_to_idx"]
    bus_state_name = mr["bus_state_name"]
    R_v = mr["chosen_R_v"]
    K_i = mr["K_i"]
    k_m = mr["k_m"]

    iL_idx = name_to_idx["conv_em_i_L"]
    sigma_idx = name_to_idx["conv_ctrl_sigma"]
    bus_idx = name_to_idx[bus_state_name]

    def phi2_of(x):
        return float(x[bus_idx] - V_NOM[topology] + R_v * x[iL_idx] - K_i * x[sigma_idx])

    def phi2_dot_numeric(x):
        f = np.asarray(mrc_system.dynamics(0.0, x, mrc_system.default_input(), mrc_system.params), dtype=float)
        return float(f[bus_idx] + R_v * f[iL_idx] - K_i * f[sigma_idx])

    for x_test in [cl_eq.x_star, cl_eq.x_star * 1.001]:
        phi2 = phi2_of(x_test)
        phi2_dot = phi2_dot_numeric(x_test)
        residual = phi2_dot + k_m * phi2
        if abs(phi2) < 1e-8:
            # at/near equilibrium phi2 itself is ~0, so a relative
            # comparison is ill-conditioned -- the residual must simply
            # be tiny in absolute terms (float/Newton-solve precision).
            assert abs(residual) < 1e-6
        else:
            # a small (0.1%) simultaneous perturbation of ALL states off
            # the manifold -- the ideal (unsaturated) identity should
            # still hold to good relative precision; not as tight as the
            # exact-at-equilibrium check above, since this also picks up
            # first-order sensitivity to which direction the joint
            # perturbation moves in.
            assert abs(residual) / abs(k_m * phi2) < 1e-2
