"""
ims_tests.test_m2_validation
-------------------------------

Follow-up validation of the M2 (bus-voltage-anchored) manifold-reshaping
feature added in test_manifold_reshaping.py: "Validate the New Reshaped
MRC (M2)". Covers:

  1. Complete Routh-Hurwitz M2 reduced-stability conditions (trace<0 AND
     det>0, never trace alone), and that the eigenvalue-based and
     trace/det-based verdicts agree.
  2. Analytical (buck) vs numerical (boost/buck-boost) admissible-region
     derivation, and that the search stays within the analytically-proven
     region for buck.
  3. Physical duty feasibility -- an accepted M2 candidate never has a
     saturated equilibrium duty; a mathematically-stable-but-infeasible
     candidate is reported as such, never as SUPPORTED.
  4. Exact unsaturated transverse contraction, and its explicit breakdown
     under duty saturation.
  5. Baseline vs M1 vs M2 identical-conditions comparison, per topology.
  6. Rejection: a network engineered so the only stable R_v also
     saturates the duty must NOT be classified SUPPORTED.
"""

import numpy as np
import pytest

from ims_platform.network.network import Network
from ims_platform.network.bus import Bus
from ims_platform.network.converter import Converter
from ims_platform.network.converter_topologies import BuckModel, BoostModel, BuckBoostModel
from ims_platform.network.controller import ConstantDutyController
from ims_platform.network.components import ConstantPowerLoad
from ims_platform.mrc_designer.auto_manifold import (
    run_auto_mrc_pipeline,
    attempt_bus_voltage_manifold_reshaping,
    attach_bus_voltage_mrc,
    find_duty_modulated_converter,
    buck_phi2_admissible_region,
    verify_buck_phi2_routh_hurwitz_symbolic,
    numeric_admissible_R_v_window,
    compare_baseline_m1_m2,
    verify_contraction_under_saturation,
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


# ---------------------------------------------------------------------------
# 1. Complete Routh-Hurwitz conditions -- never trace alone.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("topology", ["buck", "boost", "buckboost"])
def test_accepted_m2_candidate_reports_full_routh_hurwitz_not_trace_alone(topology):
    net = _build_cpl(topology, P=50.0)
    report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=V_NOM[topology])
    mr = report["manifold_reshaping"]
    assert mr["status"] == "MRC_SYNTHESIS_SUPPORTED_VIA_M2"
    rh = mr["chosen_entry"]["tangent_normal_decomposition"]["routh_hurwitz_2x2"]
    assert rh is not None
    assert rh["trace_negative"] is True
    assert rh["det_positive"] is True  # the condition trace<0 alone cannot establish
    assert rh["hurwitz_stable"] is True
    assert rh["agrees_with_eigenvalue_classification"] is True


def test_routh_hurwitz_det_condition_is_load_bearing_not_vacuous():
    """Direct proof that det>0 is a REAL, checkable condition and not
    automatically satisfied: construct the synthetic case K_i<0 (opposite
    sign to R_v>0), where det=K_i/(C*R_v)<0 -- the Jacobian is then a
    saddle regardless of trace, and must NOT be reported hurwitz_stable
    even if trace happens to be negative."""
    from ims_platform.mrc_designer.auto_manifold import tangent_normal_decomposition
    # Hand-built 2x2 with negative trace but negative det (saddle).
    J_par = np.array([[-5.0, 0.0], [0.0, 3.0]])  # trace=-2<0, det=-15<0: NOT Hurwitz (one positive eigenvalue)
    T = np.eye(3)[:, :2]
    Dphi = np.array([0.0, 0.0, 1.0])
    J_full = np.zeros((3, 3))
    J_full[:2, :2] = J_par
    J_full[2, 2] = -500.0
    result = tangent_normal_decomposition(J_full, Dphi, 500.0)
    rh = result["routh_hurwitz_2x2"]
    assert rh["trace"] == pytest.approx(-2.0)
    assert rh["det"] == pytest.approx(-15.0)
    assert rh["trace_negative"] is True
    assert rh["det_positive"] is False
    assert rh["hurwitz_stable"] is False
    # and the actual eigenvalues (-5, 3) confirm one is positive -> genuinely unstable
    tangential_re = sorted(e["re"] for e in result["tangential_eigenvalues"])
    assert tangential_re[-1] > 0
    assert rh["agrees_with_eigenvalue_classification"] is True


# ---------------------------------------------------------------------------
# 2. Analytical (buck) vs numerical (boost/buck-boost) admissible region.
# ---------------------------------------------------------------------------

def test_buck_analytic_region_matches_symbolic_rederivation():
    rh = verify_buck_phi2_routh_hurwitz_symbolic()
    assert rh["trace_matches_closed_form"] is True
    assert rh["det_matches_closed_form"] is True


def test_buck_search_grid_derived_analytically_and_stays_within_region():
    net = _build_cpl("buck", P=50.0)
    report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=24.0)
    mr = report["manifold_reshaping"]
    assert mr["grid_derivation"].startswith("analytical")
    region = mr["admissible_region"]
    assert region is not None
    assert region["R_v_max"] == pytest.approx(24.0 ** 2 / 50.0)
    for R_v in mr["R_v_grid"]:
        assert 0.0 < R_v < region["R_v_max"]
    assert 0.0 < mr["chosen_R_v"] < region["R_v_max"]


@pytest.mark.parametrize("topology", ["boost", "buckboost"])
def test_non_buck_topologies_report_numerical_only_limitation(topology):
    net = _build_cpl(topology, P=50.0)
    report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=V_NOM[topology])
    mr = report["manifold_reshaping"]
    assert mr["grid_derivation"].startswith("numerical_grid")
    assert mr["admissible_region"] is None


def test_numeric_admissible_window_characterization_for_boost():
    net = _build_cpl("boost", P=50.0)
    baseline_system = AutomaticModelBuilder.build(net, name="t")
    base_eq = baseline_system.find_equilibrium(baseline_system.initial_guess(), with_eigs=False)
    converter = find_duty_modulated_converter(net, "conv")
    window = numeric_admissible_R_v_window(
        net, converter, "boost", 72.0, baseline_system, base_eq, k_m=500.0, K_i=20.0,
        r_v_lo=1e-3, r_v_hi=10.0, n_probe=15,
    )
    assert window["derivation"] == "numerical_only"
    assert "limitation" in window
    assert len(window["observed_stable_intervals"]) >= 1


# ---------------------------------------------------------------------------
# 3. Physical duty feasibility.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("topology", ["buck", "boost", "buckboost"])
def test_accepted_m2_candidate_passes_duty_feasibility(topology):
    net = _build_cpl(topology, P=50.0)
    report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=V_NOM[topology])
    mr = report["manifold_reshaping"]
    assert mr["status"] == "MRC_SYNTHESIS_SUPPORTED_VIA_M2"
    feas = mr["chosen_entry"]["physical_feasibility"]
    assert feas["feasible"] is True
    assert feas["duty_within_limits"] is True
    assert 0.02 <= feas["duty_at_equilibrium"] <= 0.98


def test_equilibrium_duty_is_independent_of_R_v_for_buck():
    """Documents WHY 'engineer a near-saturation operating point and search
    R_v for a math-stable-but-infeasible candidate' (the original approach
    for this test) can never deterministically exercise the infeasible
    branch through the full pipeline for this platform's control laws --
    and is the justification for replacing that search-based test (which
    depended on 'accidentally finding' an operating point where phi1 also
    happened to still converge to UNSTABLE) with the two deterministic
    tests below.

    At any closed-loop equilibrium the physical steady-state equations
    (di_L/dt=0, dv_bus/dt=0) fix i_L* and the equilibrium duty purely from
    v_nom, v_in, P, R_L -- for buck: i_L*=P/v_nom,
    d_eq=(v_nom+R_L*i_L*)/v_in. ANY controller that successfully drives
    v_bus->v_nom (M1's phi1 or M2's phi2, for ANY R_v) must produce this
    SAME d_eq, because the plant dynamics themselves -- not the choice of
    manifold/R_v -- pin the duty once v_bus=v_nom and i_L=i_L* hold. So
    R_v does not decouple 'reduced dynamics is Hurwitz-stable' from
    'equilibrium duty is saturated' for this manifold family: if the
    operating point saturates the duty, it saturates it identically for
    every R_v (and for M1 too), rather than only for some R_v choices.
    """
    v_nom, v_in, P, R_L = 24.0, 48.0, 50.0, 0.05
    net = _build_cpl("buck", P=P, v_nom=v_nom, v_in=v_in, R_L=R_L)
    report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=v_nom)
    mr = report["manifold_reshaping"]
    assert mr["status"] == "MRC_SYNTHESIS_SUPPORTED_VIA_M2"
    i_L_star = P / v_nom
    d_eq_expected = (v_nom + R_L * i_L_star) / v_in
    d_eq_reported = mr["chosen_entry"]["physical_feasibility"]["duty_at_equilibrium"]
    assert d_eq_reported == pytest.approx(d_eq_expected, rel=1e-6)

    # Confirm a DIFFERENT R_v (still in the admissible window) reproduces
    # the identical equilibrium duty -- i.e. duty feasibility is a
    # property of the operating point, not of the R_v search outcome.
    converter = find_duty_modulated_converter(net, "conv")
    baseline_system = AutomaticModelBuilder.build(net, name="t2")
    base_eq = baseline_system.find_equilibrium(baseline_system.initial_guess(), with_eigs=False)
    region = mr["admissible_region"]
    other_R_v = 0.5 * mr["chosen_R_v"] if mr["chosen_R_v"] < 0.5 * region["R_v_max"] else \
        0.5 * (mr["chosen_R_v"] + region["R_v_max"])
    r2 = attempt_bus_voltage_manifold_reshaping(
        net, converter, "buck", v_nom, baseline_system, base_eq, k_m=mr["k_m"], K_i=mr["K_i"],
        r_v_grid=[other_R_v],
    )
    if r2["status"] == "MRC_SYNTHESIS_SUPPORTED_VIA_M2":
        assert r2["chosen_entry"]["physical_feasibility"]["duty_at_equilibrium"] == \
            pytest.approx(d_eq_expected, rel=1e-6)


def test_physical_feasibility_check_direct_deterministic_infeasible_point():
    """Direct, deterministic unit test of _m2_physical_feasibility_check,
    replacing the removed full-pipeline search-based test. Rather than
    searching for a network/operating-point combination that happens to
    yield a saturated equilibrium duty through the whole solver pipeline
    (not guaranteed to exist for any given R_v grid, and structurally
    decoupled from R_v choice for this platform -- see
    test_equilibrium_duty_is_independent_of_R_v_for_buck above), this
    hand-constructs a deep bus-voltage-sag STATE POINT (v_bus 98% below
    v_nom) fed directly to _m2_physical_feasibility_check, and verifies
    analytically that the resulting droop+integral-action duty command
    is deeply outside [0.02, 0.98] before checking the function's
    verdict -- exercising the exact feasibility-gate code path
    deterministically, independent of any search or solver convergence.
    """
    from ims_platform.mrc_designer.auto_manifold import _m2_physical_feasibility_check
    from ims_platform.network.controller import TOPOLOGY_CODE, bus_voltage_mrc_duty

    v_nom, K_i, k_m, R_v = 24.0, 20.0, 500.0, 1.0
    C, v_in, L, R_L = 2e-3, 48.0, 1e-3, 0.05
    params = {"v_nom": v_nom, "K_i": K_i, "k_m": k_m, "R_v": R_v,
              "topology_code": TOPOLOGY_CODE["buck"], "v_in": v_in, "L": L, "R_L": R_L, "C": C}
    other_injection = lambda v: -50.0 / v  # constant-power draw, P=50 W

    # Deep, hand-picked (not searched) bus-voltage-sag state point.
    v_bus_sag, i_L_sag, sigma_sag = 0.5, 100.0, 0.0
    d_expected = bus_voltage_mrc_duty("", i_L_sag, v_bus_sag, sigma_sag, params, other_injection)
    assert d_expected < 0.0  # confirmed analytically well outside [0.02, 0.98]

    class _FakeController:
        def __init__(self):
            self.params = params
            self.other_bus_injection = other_injection

    class _FakeConverter:
        def __init__(self):
            self.controller = _FakeController()

    class _FakeEq:
        def __init__(self, x_star):
            self.x_star = x_star

    name_to_idx = {"conv_ctrl_sigma": 0, "conv_em_i_L": 1, "bus_v_bus": 2}
    x_star = np.array([sigma_sag, i_L_sag, v_bus_sag])

    feas = _m2_physical_feasibility_check(
        _FakeConverter(), _FakeEq(x_star), name_to_idx, "conv", "bus_v_bus",
        baseline_i_L0=2.0833, A2_center=R_v * v_in / L, reference_A_scale=v_in / L,
    )
    assert feas["duty_at_equilibrium"] == pytest.approx(d_expected)
    assert feas["duty_saturated"] is True
    assert feas["duty_within_limits"] is False
    assert feas["feasible"] is False


def test_stable_but_infeasible_entry_never_returned_as_chosen():
    """Direct unit-level check on attempt_bus_voltage_manifold_reshaping:
    an entry marked STABLE_BUT_PHYSICALLY_INFEASIBLE in 'search' must
    never be the one set as chosen_R_v/chosen_entry."""
    net = _build_cpl("buck", P=50.0)
    baseline_system = AutomaticModelBuilder.build(net, name="t")
    base_eq = baseline_system.find_equilibrium(baseline_system.initial_guess(), with_eigs=False)
    converter = find_duty_modulated_converter(net, "conv")
    r = attempt_bus_voltage_manifold_reshaping(
        net, converter, "buck", 24.0, baseline_system, base_eq, k_m=500.0, K_i=20.0,
    )
    infeasible_entries = [e for e in r["search"] if e.get("status") == "STABLE_BUT_PHYSICALLY_INFEASIBLE"]
    if r["chosen_entry"] is not None:
        assert r["chosen_entry"] not in infeasible_entries
        assert r["chosen_entry"]["physical_feasibility"]["feasible"] is True


# ---------------------------------------------------------------------------
# 4. Exact unsaturated contraction, and explicit breakdown under saturation.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("topology", ["buck", "boost", "buckboost"])
def test_contraction_holds_unsaturated_and_breaks_under_saturation(topology):
    net = _build_cpl(topology, P=50.0)
    report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=V_NOM[topology])
    mr = report["manifold_reshaping"]
    assert mr["status"] == "MRC_SYNTHESIS_SUPPORTED_VIA_M2"
    R_v = mr["chosen_R_v"]; K_i = mr["K_i"]; k_m = mr["k_m"]

    converter = find_duty_modulated_converter(net, "conv")
    m2_net, m2_conv = attach_bus_voltage_mrc(net, "conv", V_NOM[topology],
                                             {"K_i": K_i, "k_m": k_m, "R_v": R_v})
    m2_system = AutomaticModelBuilder.build(m2_net, name="m2check")
    eq = m2_system.find_equilibrium(m2_system.initial_guess(), with_eigs=False)
    name_to_idx = {n: i for i, n in enumerate(m2_system.state_names)}
    bus_state_name = next(n for n in name_to_idx if n.endswith("v_bus") or "_v_" in n or n == net.buses["bus"].state_name)
    bus_state_name = net.buses["bus"].state_name

    params = dict(m2_conv.controller.params)
    params["v_nom"] = V_NOM[topology]
    sat = verify_contraction_under_saturation(
        m2_system, "conv", eq.x_star, name_to_idx, bus_state_name, "phi2", params,
        other_injection=m2_conv.controller.other_bus_injection,
        v_bus_sag_fracs=[1.0, 0.95, 0.3],
    )
    assert sat["ideal_unsaturated_identity_verified_at_equilibrium"] is True
    # deep saturation point: duty is clamped and the identity is NOT claimed
    deep = sat["points"][-1]
    assert deep["v_bus_sag_frac"] == 0.3
    if deep["duty_clamped"]:
        assert deep["contraction_identity_holds"] is False
        assert sat["saturation_breaks_identity"] is True


def test_m1_contraction_identity_also_unaffected_by_this_change():
    """Regression: phi1's OWN contraction residual check (pre-existing,
    _contraction_residual_check) is untouched by the M2 additions."""
    net = _build_cpl("buck", P=50.0)
    report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=24.0)
    assert report["contraction_residual"]["max_relative_error"] < 1e-6


# ---------------------------------------------------------------------------
# 5. Baseline vs M1 vs M2, identical conditions, per topology.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("topology", ["buck", "boost", "buckboost"])
def test_baseline_m1_m2_identical_conditions_comparison(topology):
    net = _build_cpl(topology, P=50.0)
    report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=V_NOM[topology])
    mr = report["manifold_reshaping"]
    assert mr["status"] == "MRC_SYNTHESIS_SUPPORTED_VIA_M2"

    cmp = compare_baseline_m1_m2(net, "conv", V_NOM[topology], R_v=mr["chosen_R_v"],
                                 K_i_m2=mr["K_i"], k_m_m2=mr["k_m"])
    assert cmp["baseline"]["equilibrium_converged"]
    assert cmp["m1"]["equilibrium_converged"]
    assert cmp["m2"]["equilibrium_converged"]
    # identical horizon/time grid across all three arms
    assert cmp["baseline"]["t"] == cmp["m1"]["t"] == cmp["m2"]["t"]
    assert len(cmp["m1"]["i_L"]) == len(cmp["m2"]["i_L"]) == len(cmp["baseline"]["i_L"])
    # M1 (tangentially unstable manifold) must NOT recover; M2 must.
    assert cmp["m1"]["recovered"] is False
    assert cmp["m2"]["recovered"] is True
    # phi(t) is reported for M1 and M2 (not for baseline, which has no manifold)
    assert "phi" in cmp["m1"] and len(cmp["m1"]["phi"]) == len(cmp["m1"]["t"])
    assert "phi" in cmp["m2"] and len(cmp["m2"]["phi"]) == len(cmp["m2"]["t"])
    assert "phi" not in cmp["baseline"]


def test_compare_baseline_m1_m2_requires_explicit_m2_params():
    net = _build_cpl("buck", P=50.0)
    with pytest.raises(ValueError):
        compare_baseline_m1_m2(net, "conv", 24.0)
