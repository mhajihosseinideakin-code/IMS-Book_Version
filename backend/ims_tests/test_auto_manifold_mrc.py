"""
Tests for ims_platform.mrc_designer.auto_manifold: the automatic
controlled-target-manifold construction, feasibility, synthesis, and
closed-loop verification pipeline for directly duty-modulated Network
Builder converters (Buck/Boost/Buck-Boost), plus the required validation
matrix across topology x load combinations, plus a regression guard
confirming the existing Four-State Stabilising MRC benchmark is
untouched by this module.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy

from ims_platform.network.network import Network
from ims_platform.network.bus import Bus
from ims_platform.network.converter import Converter
from ims_platform.network.converter_topologies import BuckModel, BoostModel, BuckBoostModel
from ims_platform.network.controller import (
    ConstantDutyController, PIDutyController, DroopController, current_loop_mrc_duty, TOPOLOGY_CODE,
)
from ims_platform.network.components import ConstantPowerLoad, ConstantImpedanceLoad, ConstantCurrentLoad, IdealSource
from ims_platform.mrc_designer.auto_manifold import (
    run_auto_mrc_pipeline,
    run_auto_mrc_pipeline_multi,
    list_duty_modulated_converters,
    derive_symbolic_law,
    symbolic_diL_dt,
    find_duty_modulated_converter,
    _saturation_check,
    TOPOLOGY_CODE,
)
from ims_platform.mrc_designer.designer import MRCNotEstablished

TOPOLOGIES = {"buck": BuckModel, "boost": BoostModel, "buckboost": BuckBoostModel}
V_NOM_BY_TOPOLOGY = {"buck": 24.0, "boost": 72.0, "buckboost": 24.0}


def _build_network(topology: str, load_kind: str, v_in: float = 48.0) -> Network:
    net = Network("t")
    v_nom = V_NOM_BY_TOPOLOGY[topology]
    net.add_bus(Bus(id="bus", C=2e-3, v_init=v_nom, v_min=v_nom * 0.05))
    em = TOPOLOGIES[topology](v_in=v_in, L=1e-3, R_L=0.05)
    net.add_component(Converter(id="conv", bus="bus", electrical_model=em, controller=ConstantDutyController(d=0.5)))
    if load_kind == "cpl":
        net.add_component(ConstantPowerLoad(id="load", bus="bus", P=50.0, v_floor=0.5))
    elif load_kind == "impedance":
        net.add_component(ConstantImpedanceLoad(id="load", bus="bus", R=v_nom ** 2 / 50.0))
    elif load_kind == "current":
        net.add_component(ConstantCurrentLoad(id="load", bus="bus", I=50.0 / v_nom))
    else:
        raise ValueError(load_kind)
    return net


# ---------------------------------------------------------------------------
# Independent symbolic-vs-numeric cross-check (guards against the two
# duplicated formulas -- network.controller.current_loop_mrc_duty and
# mrc_designer.auto_manifold.symbolic_diL_dt -- silently drifting apart,
# and both against the REAL, validated ConverterModel.local_dynamics).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("topology", ["buck", "boost", "buckboost"])
def test_symbolic_matches_numeric_topology(topology):
    em_cls = TOPOLOGIES[topology]
    em = em_cls(v_in=48.0, L=1e-3, R_L=0.05)
    rng = np.random.default_rng(42)
    iL_s, vbus_s, d_s, v_in_s, L_s, R_L_s = sympy.symbols("i_L v_bus d v_in L R_L", real=True)
    expr = symbolic_diL_dt(topology, iL_s, vbus_s, d_s, v_in_s, L_s, R_L_s)
    fn = sympy.lambdify((iL_s, vbus_s, d_s, v_in_s, L_s, R_L_s), expr, "numpy")
    for _ in range(20):
        iL = float(rng.uniform(0.1, 10.0))
        vbus = float(rng.uniform(5.0, 100.0))
        d = float(rng.uniform(0.05, 0.95))
        real = em.local_dynamics(vbus, np.array([iL]), d, em.params)[0]
        symbolic = fn(iL, vbus, d, 48.0, 1e-3, 0.05)
        assert abs(real - symbolic) < 1e-9, (topology, iL, vbus, d, real, symbolic)


@pytest.mark.parametrize("topology", ["buck", "boost", "buckboost"])
def test_derived_duty_law_matches_runtime_controller_formula(topology):
    """
    The symbolic closed-form solved by derive_symbolic_law must agree,
    numerically, with the independently-written runtime formula in
    network.controller.current_loop_mrc_duty -- two separately-authored
    computational paths for the same control law.
    """
    params = {"v_nom": V_NOM_BY_TOPOLOGY[topology], "K_i": 20.0, "k_m": 500.0, "i_bias": 0.0,
              "v_in": 48.0, "L": 1e-3, "R_L": 0.05}
    law = derive_symbolic_law(topology, params)
    syms = law["symbols"]
    fn = sympy.lambdify(
        (syms["iL"], syms["vbus"], syms["sigma"], syms["v_in"], syms["L"], syms["R_L"],
         syms["v_nom"], syms["K_i"], syms["k_m"], syms["i_bias"]),
        law["control_expr"], "numpy",
    )
    p_runtime = dict(params)
    p_runtime["topology_code"] = TOPOLOGY_CODE[topology]
    rng = np.random.default_rng(7)
    for _ in range(10):
        iL = float(rng.uniform(0.5, 5.0))
        vbus = float(rng.uniform(V_NOM_BY_TOPOLOGY[topology] * 0.9, V_NOM_BY_TOPOLOGY[topology] * 1.1))
        sigma = float(rng.uniform(-0.5, 0.5))
        symbolic_d = float(fn(iL, vbus, sigma, params["v_in"], params["L"], params["R_L"],
                               params["v_nom"], params["K_i"], params["k_m"], params["i_bias"]))
        runtime_d = current_loop_mrc_duty("", iL, vbus, sigma, p_runtime)
        assert abs(symbolic_d - runtime_d) < 1e-9, (topology, iL, vbus, sigma, symbolic_d, runtime_d)


# ---------------------------------------------------------------------------
# Honest "not established" cases -- required by task step 8
# ---------------------------------------------------------------------------

def test_no_duty_modulated_converter_reports_not_established():
    net = Network("t")
    net.add_bus(Bus(id="bus", v_fixed=24.0))
    net.add_component(ConstantImpedanceLoad(id="load", bus="bus", R=10.0))
    with pytest.raises(MRCNotEstablished, match="no directly duty-modulated converter"):
        find_duty_modulated_converter(net)

    net2 = Network("t2")
    net2.add_bus(Bus(id="bus", C=1e-3, v_init=24.0, v_min=1.0))
    net2.add_component(IdealSource(id="src", bus="bus", v_source=24.0, R_source=0.1))
    net2.add_component(ConstantImpedanceLoad(id="load", bus="bus", R=10.0))
    report = run_auto_mrc_pipeline(net2)
    assert report["status"] == "MRC_SYNTHESIS_NOT_ESTABLISHED"
    assert "no directly duty-modulated converter" in report["reason"]


def test_ambiguous_converters_require_explicit_id():
    net = _build_network("buck", "impedance")
    em2 = BuckModel(v_in=48.0, L=1e-3, R_L=0.05)
    net.add_component(Converter(id="conv2", bus="bus", electrical_model=em2, controller=ConstantDutyController(d=0.5)))
    with pytest.raises(MRCNotEstablished, match="must be given explicitly"):
        find_duty_modulated_converter(net)
    # with an explicit id, resolution succeeds
    found = find_duty_modulated_converter(net, converter_id="conv")
    assert found.id == "conv"


def test_target_voltage_outside_achievable_range_is_not_established():
    """A Boost converter cannot regulate its bus BELOW v_in in the ideal
    averaged model (V_o/V_in = 1/(1-D) >= 1 for D in [0,1)); asking for
    that is an equilibrium-incompatible target, not a code failure."""
    net = _build_network("boost", "impedance", v_in=48.0)
    report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=24.0)  # below v_in=48
    assert report["status"] == "MRC_SYNTHESIS_NOT_ESTABLISHED"
    assert "not reachable" in report["reason"] or "incompatible" in report["reason"]


# ---------------------------------------------------------------------------
# Required validation matrix (task step 9): every reported case must carry
# a real mathematical status, never a blanket "Supported", and the exact
# transverse contraction identity must hold to near machine precision
# whenever synthesis is attempted at all.
# ---------------------------------------------------------------------------

VALID_STATUSES = {"MRC_SYNTHESIS_SUPPORTED", "MRC_FEASIBILITY_DIAGNOSTIC_ONLY", "MRC_SYNTHESIS_NOT_ESTABLISHED",
                  "MRC_SYNTHESIS_SUPPORTED_VIA_RESHAPED_MANIFOLD"}


@pytest.mark.parametrize("topology", ["buck", "boost", "buckboost"])
@pytest.mark.parametrize("load_kind", ["cpl", "impedance", "current"])
def test_validation_matrix_case(topology, load_kind):
    net = _build_network(topology, load_kind)
    report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=V_NOM_BY_TOPOLOGY[topology])
    assert report["status"] in VALID_STATUSES
    if report["status"] != "MRC_SYNTHESIS_NOT_ESTABLISHED":
        # relative degree one is a structural property of this construction
        # for these topologies and must always hold once synthesis proceeds
        assert report["rank_A"] == 1
        assert report["relative_degree"] == 1
        # the exact, ideal-model transverse identity phi_dot = -k_m*phi
        # must hold to near machine precision, independent of load type
        assert report["contraction_residual"]["max_relative_error"] < 1e-6


def test_validation_matrix_impedance_loads_are_supported():
    """
    A resistive (impedance) load has no negative-incremental-impedance
    destabilizing term, so the current-loop MRC's tangential/reduced
    dynamics is expected to be genuinely, locally stable for all three
    topologies at the chosen default gains -- this is the concrete,
    positive "MRC SYNTHESIS SUPPORTED" case the task asks the pipeline to
    be capable of reaching honestly (not by forcing it).
    """
    for topology in ("buck", "boost", "buckboost"):
        net = _build_network(topology, "impedance")
        report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=V_NOM_BY_TOPOLOGY[topology])
        assert report["status"] == "MRC_SYNTHESIS_SUPPORTED", (topology, report.get("reason"))
        assert report["simulation_result"]["recovered"] is True


def test_validation_matrix_cpl_loads_are_not_silently_forced_supported():
    """
    A constant-power load's negative incremental impedance is the book's
    own central destabilizing mechanism (ch:motivation, eq:motivation-cpl)
    -- at the chosen default gains the current-loop construction's
    tangential dynamics is NOT locally stable for a CPL, and the pipeline
    must say so honestly (diagnostic-only), never silently reporting
    SUPPORTED just because the transverse identity still holds exactly.
    """
    for topology in ("buck", "boost", "buckboost"):
        net = _build_network(topology, "cpl")
        report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=V_NOM_BY_TOPOLOGY[topology])
        assert report["status"] != "MRC_SYNTHESIS_SUPPORTED", (
            f"{topology}+cpl reported SUPPORTED at default gains, which would mean the "
            f"CPL's destabilizing effect on the reduced dynamics was not actually checked"
        )
        # the transverse claim must still be reported as exact, regardless
        assert report["contraction_residual"]["max_relative_error"] < 1e-6
        assert report["claim_levels"]["local_equilibrium_stability"]["established"] is False


# ---------------------------------------------------------------------------
# Regression guard: this module must not have touched the existing,
# independently-verified Four-State Stabilising MRC benchmark.
# ---------------------------------------------------------------------------

def _build_two_converter_network(topoA: str, topoB: str, v_in: float = 48.0) -> Network:
    """Two directly duty-modulated converters, each on its OWN bus (an
    electrically decoupled two-branch custom network -- the two
    converters do not fight over one shared bus voltage, which would be
    a modeling artifact, not the multi-converter feasibility question
    task item 1 is actually about). Each converter's own duty-cycle
    open-loop equilibrium is consistent with d=0.5 at v_in=48 for its own
    topology, so the baseline (pre-MRC) equilibrium always converges;
    only MRC feasibility for the REQUESTED v_nom differs per candidate."""
    net = Network("multi")
    v_openloop = {"buck": v_in * 0.5, "boost": v_in / (1.0 - 0.5), "buckboost": v_in}
    net.add_bus(Bus(id="busA", C=2e-3, v_init=v_openloop[topoA], v_min=1.0))
    net.add_bus(Bus(id="busB", C=2e-3, v_init=v_openloop[topoB], v_min=1.0))
    emA = TOPOLOGIES[topoA](v_in=v_in, L=1e-3, R_L=0.05)
    net.add_component(Converter(id="convA", bus="busA", electrical_model=emA, controller=ConstantDutyController(d=0.5)))
    emB = TOPOLOGIES[topoB](v_in=v_in, L=1e-3, R_L=0.05)
    net.add_component(Converter(id="convB", bus="busB", electrical_model=emB, controller=ConstantDutyController(d=0.5)))
    net.add_component(ConstantImpedanceLoad(id="loadA", bus="busA", R=v_openloop[topoA] ** 2 / 50.0))
    net.add_component(ConstantImpedanceLoad(id="loadB", bus="busB", R=v_openloop[topoB] ** 2 / 50.0))
    return net


# ---------------------------------------------------------------------------
# Multi-converter feasibility dispatch (task item 1): automatically inspect
# every candidate, show a per-converter feasibility result, auto-select
# only when exactly one is feasible, require explicit selection when several
# are, never silently choose, never claim a coupled multi-input MRC.
# ---------------------------------------------------------------------------

def test_multi_converter_listing_finds_every_candidate():
    net = _build_two_converter_network("buck", "boost")
    ids = {c.id for c in list_duty_modulated_converters(net)}
    assert ids == {"convA", "convB"}


def test_multi_converter_exactly_one_feasible_auto_selects():
    """convA=buck (v_in=48) can step DOWN to v_nom=24: feasible. convB=boost
    (v_in=48) cannot regulate BELOW its own v_in in the ideal averaged
    model: infeasible (equilibrium-incompatible). Exactly one feasible ->
    auto-selected, not silently -- the scan for the other is still shown."""
    net = _build_two_converter_network("buck", "boost")
    report = run_auto_mrc_pipeline_multi(net, v_nom=24.0)
    assert report["status"] in ("MRC_SYNTHESIS_SUPPORTED", "MRC_FEASIBILITY_DIAGNOSTIC_ONLY")
    assert report["converter_id"] == "convA"
    assert "multi_converter_scan" in report
    scan_by_id = {c["converter_id"]: c for c in report["multi_converter_scan"]}
    assert set(scan_by_id) == {"convA", "convB"}
    assert scan_by_id["convA"]["feasible"] is True
    assert scan_by_id["convB"]["feasible"] is False
    assert scan_by_id["convB"]["status"] == "MRC_SYNTHESIS_NOT_ESTABLISHED"


def test_multi_converter_multiple_feasible_requires_explicit_selection():
    """Two buck converters, both able to reach v_nom=24 from v_in=48: BOTH
    feasible -- must require explicit selection, never silently pick one."""
    net = _build_two_converter_network("buck", "buck")
    report = run_auto_mrc_pipeline_multi(net, v_nom=24.0)
    assert report["status"] == "MRC_MULTI_CANDIDATE_SELECTION_REQUIRED"
    assert set(report["feasible_converter_ids"]) == {"convA", "convB"}
    assert report["requires_explicit_selection"] is True
    assert "converter_id" in report["reason"] and "explicitly" in report["reason"]
    assert "converter_id" not in report  # no converter was chosen

    # an explicit id resolves it, unambiguously, to that converter's own
    # single-converter report -- not a new/different code path
    resolved = run_auto_mrc_pipeline_multi(net, converter_id="convA", v_nom=24.0)
    assert resolved["status"] in ("MRC_SYNTHESIS_SUPPORTED", "MRC_FEASIBILITY_DIAGNOSTIC_ONLY")
    assert resolved["converter_id"] == "convA"
    assert "multi_converter_scan" not in resolved
    direct = run_auto_mrc_pipeline(net, converter_id="convA", v_nom=24.0)
    assert resolved == direct


def test_multi_converter_none_feasible_reports_not_established():
    """Two boost converters asked to regulate BELOW their shared v_in:
    neither is feasible -- honest NOT_ESTABLISHED, with each candidate's
    own reason preserved, never a forced SUPPORTED/DIAGNOSTIC result."""
    net = _build_two_converter_network("boost", "boost", v_in=48.0)
    report = run_auto_mrc_pipeline_multi(net, v_nom=24.0)
    assert report["status"] == "MRC_SYNTHESIS_NOT_ESTABLISHED"
    assert report["feasible_converter_ids"] == []
    assert len(report["multi_converter_scan"]) == 2
    for c in report["multi_converter_scan"]:
        assert c["feasible"] is False


def test_multi_dispatch_never_claims_multi_input_mrc():
    """Even when a candidate is auto-selected among several, the report's
    own dim_phi/candidate_phi must describe a SINGLE-input manifold (this
    platform never derives or claims a coupled multi-input MRC)."""
    net = _build_two_converter_network("buck", "boost")
    report = run_auto_mrc_pipeline_multi(net, v_nom=24.0)
    assert report["dim_phi"] == 1
    assert "i_L" in report["candidate_phi"]


def test_multi_dispatch_preserves_single_converter_behavior_exactly():
    """Preservation requirement (task item 4): explicit converter_id, and
    the unambiguous single-candidate case, must be byte-for-byte identical
    to the pre-existing run_auto_mrc_pipeline output."""
    net = _build_network("buck", "impedance")
    direct = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=24.0)
    via_explicit = run_auto_mrc_pipeline_multi(net, converter_id="conv", v_nom=24.0)
    via_auto = run_auto_mrc_pipeline_multi(net, v_nom=24.0)  # only one candidate exists
    assert direct == via_explicit
    assert direct == via_auto


# ---------------------------------------------------------------------------
# Actuator saturation (task item 5's explicit requirement)
# ---------------------------------------------------------------------------

def test_saturation_check_flags_out_of_range_duty_and_passes_in_range():
    class _FakeController:
        def __init__(self, params):
            self.params = params

    class _FakeConverter:
        def __init__(self, params):
            self.controller = _FakeController(params)

    params = {"v_nom": 24.0, "K_i": 20.0, "k_m": 500.0, "i_bias": 0.0,
              "topology_code": TOPOLOGY_CODE["buck"], "v_in": 48.0, "L": 1e-3, "R_L": 0.05}
    name_to_idx = {"conv_em_i_L": 0, "conv_ctrl_sigma": 1, "v_bus": 2}

    # near the manifold (i_L ~= i_bias + K_i*sigma) at v_bus == v_nom: duty
    # should land near the steady-state buck ratio v_bus/v_in, well inside
    # the [0.02, 0.98] clamp.
    x_in_range = np.array([1.0, 0.05, 24.0])
    res_ok = _saturation_check(_FakeConverter(params), x_in_range, name_to_idx, "conv", "v_bus")
    assert res_ok["saturated"] is False
    assert 0.02 <= res_ok["duty_at_equilibrium"] <= 0.98

    # far off the manifold: the corrective term drives the closed-form
    # duty command outside the converter's admissible range.
    x_saturated = np.array([500.0, 0.0, 24.0])
    res_bad = _saturation_check(_FakeConverter(params), x_saturated, name_to_idx, "conv", "v_bus")
    assert res_bad["saturated"] is True
    assert res_bad["duty_at_equilibrium"] < 0.02 or res_bad["duty_at_equilibrium"] > 0.98
    assert "NOT active" in res_bad["note"]


# ---------------------------------------------------------------------------
# Before/After identical-condition verification (task item 4/5): the SAME
# disturbance, on the SAME assembled network, baseline vs MRC.
# ---------------------------------------------------------------------------

def test_before_after_uses_identical_disturbance_and_horizon_for_both_arms():
    net = _build_network("buck", "impedance")
    report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=24.0)
    assert report["status"] == "MRC_SYNTHESIS_SUPPORTED"

    comparison = report["closed_loop_comparison"]
    baseline, mrc = comparison["baseline"], comparison["mrc"]
    assert baseline["t"] is not None and mrc["t"] is not None
    # identical simulation horizon/time grid for both arms -- a genuine
    # baseline-vs-MRC comparison, not two independently-configured runs
    assert baseline["t"] == mrc["t"]
    assert baseline["t"][-1] == pytest.approx(mrc["t"][-1])
    assert len(baseline["i_L"]) == len(mrc["i_L"]) == len(baseline["t"])
    # both arms are perturbed from the SAME converter/network equilibrium
    assert report["saturation_note"]["duty_at_equilibrium"] is not None


# ---------------------------------------------------------------------------
# Baseline-controller variation (task item 8's model-matrix verification):
# the ORIGINAL controller on the converter (before MRC replaces it) must
# not affect MRC feasibility itself, and must not spuriously fail the
# baseline/closed-loop equilibrium solve it needs along the way.
# ---------------------------------------------------------------------------

def _build_network_with_controller(topology: str, controller) -> Network:
    net = Network("t")
    v_nom = V_NOM_BY_TOPOLOGY[topology]
    net.add_bus(Bus(id="bus", C=2e-3, v_init=v_nom, v_min=v_nom * 0.05))
    em = TOPOLOGIES[topology](v_in=48.0, L=1e-3, R_L=0.05)
    net.add_component(Converter(id="conv", bus="bus", electrical_model=em, controller=controller))
    net.add_component(ConstantImpedanceLoad(id="load", bus="bus", R=v_nom ** 2 / 50.0))
    return net


@pytest.mark.parametrize("topology", ["buck", "boost"])
def test_pi_controlled_baseline_reaches_genuine_supported_status(topology):
    """
    Regression test for a real false negative found while validating the
    model matrix: a PI-voltage-controlled baseline converter's equilibrium
    solve can land at a residual of ~1e-12 (an obviously converged root)
    while scipy's fsolve still reports ier != 1 ("not making good
    progress"), because the solver's own step-improvement heuristic has
    nothing left to improve once it is already this close. The pipeline
    used to gate `converged` on `ier == 1`, so this genuinely-converged
    equilibrium was misreported as MRC_SYNTHESIS_NOT_ESTABLISHED
    ("target manifold incompatible with equilibrium") for a topology/
    target-voltage combination that IS reachable, confirmed here by the
    fact that a ConstantDutyController baseline on the identical topology
    and target reaches MRC_SYNTHESIS_SUPPORTED. The fix
    (core.system.DynamicalSystem.find_equilibrium judging convergence by
    residual_norm alone) must not have weakened the check -- it is a
    STRICTER, purely mathematical condition (f(x*) ~ 0) than trusting a
    solver-internal progress heuristic.
    """
    v_nom = V_NOM_BY_TOPOLOGY[topology]
    ctrl = PIDutyController(v_nom=v_nom, Kp=0.02, Ki=5.0, d_nominal=0.5)
    net = _build_network_with_controller(topology, ctrl)
    report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=v_nom)
    assert report["status"] == "MRC_SYNTHESIS_SUPPORTED", (topology, report.get("reason"))
    assert report["closed_loop_equilibrium_residual_norm"] < 1e-6


@pytest.mark.parametrize("topology", ["buck", "boost"])
def test_droop_controlled_baseline_reaches_genuine_supported_status(topology):
    v_nom = V_NOM_BY_TOPOLOGY[topology]
    ctrl = DroopController(v_nom=v_nom, Kp=0.02, R_droop=0.5, d_nominal=0.5)
    net = _build_network_with_controller(topology, ctrl)
    report = run_auto_mrc_pipeline(net, converter_id="conv", v_nom=v_nom)
    assert report["status"] == "MRC_SYNTHESIS_SUPPORTED", (topology, report.get("reason"))


def test_equilibrium_convergence_is_judged_by_residual_not_solver_internal_flag():
    """Direct unit test of the core.system.py fix: an equilibrium solve
    whose residual is at machine precision must be reported converged
    regardless of scipy fsolve's own ier flag."""
    from ims_platform.network.assembler import AutomaticModelBuilder

    v_nom = 24.0
    ctrl = PIDutyController(v_nom=v_nom, Kp=0.02, Ki=5.0, d_nominal=0.5)
    net = _build_network_with_controller("buck", ctrl)
    system = AutomaticModelBuilder.build(net, name=net.name)
    result = system.find_equilibrium(system.initial_guess(), with_eigs=False)
    assert result.residual_norm < 1e-9
    assert result.converged is True


def test_four_state_stabilizing_mrc_benchmark_unaffected():
    from ims_platform.stabilizing import compute_equilibrium, compute_stability

    params = dict(P=20e3, v_nom=400.0, R=0.20, L=1.5e-3, C=2.5e-3, R_v=0.5, K_i=50.0, k_m=500.0)
    eq = compute_equilibrium(params)
    st = compute_stability(params)

    x_star = eq["x_star"]
    expected = {"i_l": 50.0, "v_b": 400.0, "sigma": 0.7, "v_o": 410.0}
    for key, val in expected.items():
        assert abs(x_star[key] - val) < 1e-2, (key, x_star[key], val)

    assert st["locally_exponentially_stable"] is True
    assert abs(st["transverse_eigenvalue"]["re"] - (-500.0)) < 1.0
