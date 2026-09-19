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
from ims_platform.network.controller import ConstantDutyController, current_loop_mrc_duty, TOPOLOGY_CODE
from ims_platform.network.components import ConstantPowerLoad, ConstantImpedanceLoad, ConstantCurrentLoad, IdealSource
from ims_platform.mrc_designer.auto_manifold import (
    run_auto_mrc_pipeline,
    derive_symbolic_law,
    symbolic_diL_dt,
    find_duty_modulated_converter,
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

VALID_STATUSES = {"MRC_SYNTHESIS_SUPPORTED", "MRC_FEASIBILITY_DIAGNOSTIC_ONLY", "MRC_SYNTHESIS_NOT_ESTABLISHED"}


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
