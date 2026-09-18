"""
Tests for the network representation and Automatic Model Builder
(architecture document Part IV Section 4.1-4.4).

The most important test in this file is
test_assembled_two_bus_network_matches_hand_built_model_exactly, which
proves -- numerically, at many random points, to floating-point precision
-- that the automatically assembled network reproduces a previously
validated hand-built model exactly. This is the network-layer analogue
of the symbolic-equality proof used to validate MRC synthesis: the
Automatic Model Builder is not merely plausible, it is checked against a
known-correct reference.
"""
import numpy as np

try:
    import pytest
except ImportError:
    pytest = None

from ims_platform.network import (
    Network, Bus, Line, ConstantPowerLoad, ConstantImpedanceLoad,
    IdealSource, AutomaticModelBuilder, NetworkValidationError,
)
from ims_platform.models import DCMicrogridCPL
from ims_platform.core import Simulator
from ims_platform.ims import IntrinsicManifold, RecoverabilityAnalyzer


def _build_two_bus_network():
    net = Network("two_bus_dc_cpl")
    net.add_bus(Bus(id="source", v_fixed=1.2))
    net.add_bus(Bus(id="load", C=0.05, v_init=1.13, v_min=0.3))
    net.add_line(Line(id="line1", from_bus="source", to_bus="load", R=0.15, L=0.005))
    net.add_component(ConstantPowerLoad(id="cpl1", bus="load", P=0.5, v_floor=0.05))
    return net


def test_bus_requires_v_fixed_if_no_capacitance():
    if pytest is not None:
        with pytest.raises(ValueError):
            Bus(id="bad")
    else:
        try:
            Bus(id="bad")
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_network_rejects_line_to_unknown_bus():
    net = Network()
    net.add_bus(Bus(id="a", v_fixed=1.0))
    if pytest is not None:
        with pytest.raises(NetworkValidationError):
            net.add_line(Line(id="l1", from_bus="a", to_bus="ghost", R=0.1, L=0.001))
    else:
        try:
            net.add_line(Line(id="l1", from_bus="a", to_bus="ghost", R=0.1, L=0.001))
            raise AssertionError("expected NetworkValidationError")
        except NetworkValidationError:
            pass


def test_network_rejects_isolated_dynamic_bus():
    net = Network()
    net.add_bus(Bus(id="a", v_fixed=1.0))
    net.add_bus(Bus(id="isolated", C=0.01, v_init=1.0))  # no branch or component attached
    if pytest is not None:
        with pytest.raises(NetworkValidationError):
            net.validate()
    else:
        try:
            net.validate()
            raise AssertionError("expected NetworkValidationError")
        except NetworkValidationError:
            pass


def test_assembled_two_bus_network_matches_hand_built_model_exactly():
    net = _build_two_bus_network()
    assembled = AutomaticModelBuilder.build(net)
    ref = DCMicrogridCPL()

    rng = np.random.default_rng(0)
    max_diff = 0.0
    for _ in range(200):
        i = rng.uniform(-2, 5)
        v = rng.uniform(0.3, 2.0)
        P = rng.uniform(0.1, 1.0)
        f_assembled = assembled.dynamics(0.0, np.array([v, i]), np.array([P]), assembled.params)
        f_ref = ref.dynamics(0.0, np.array([i, v]), np.array([P]), ref.params)
        # assembled order [dv, di]; reference order [di, dv]
        diff = np.abs(np.array([f_assembled[1], f_assembled[0]]) - f_ref)
        max_diff = max(max_diff, float(diff.max()))

    assert max_diff < 1e-10, f"assembled network diverges from hand-built reference by {max_diff}"


def test_assembled_network_equilibrium_matches_known_result():
    """The assembled network's equilibrium should match the independently validated DCMicrogridCPL result."""
    net = _build_two_bus_network()
    assembled = AutomaticModelBuilder.build(net)
    eq = assembled.find_equilibrium(assembled.initial_guess(), u=np.array([0.5]))
    assert eq.converged
    assert eq.is_stable
    # Known result (validated independently earlier in the project): v* ~ 1.1339, i* ~ 0.4409
    v_star = eq.x_star[assembled.state_names.index("v_load")]
    i_star = eq.x_star[assembled.state_names.index("i_line1")]
    assert abs(v_star - 1.1339) < 1e-3
    assert abs(i_star - 0.4409) < 1e-3


def test_full_pipeline_runs_unmodified_on_assembled_system():
    """
    The central architectural claim of Part IV: equilibrium solving,
    manifold tracing, and recoverability assessment require zero changes
    to consume an assembled network instead of a hand-written model.
    """
    net = _build_two_bus_network()
    assembled = AutomaticModelBuilder.build(net)
    x0 = assembled.initial_guess()

    eq = assembled.find_equilibrium(x0, u=np.array([0.5]))
    assert eq.converged

    M = IntrinsicManifold(assembled, param_name="cpl1_P").build(
        alpha_range=(0.05, 1.2), n_points=25, x0_guess=x0, keep_unstable=False
    )
    assert len(M.points) > 15

    sim = Simulator(assembled, method="RK45")
    analyzer = RecoverabilityAnalyzer(assembled, M, sim, recovery_tol=0.1)
    report = analyzer.assess(eq.x_star, radius=0.3, n_samples=20, t_horizon=2.0, u=np.array([0.5]))
    assert 0.0 <= report.recoverability_index <= 1.0


def test_three_bus_network_assembles_and_solves():
    """A genuinely new topology neither hand-built model can represent."""
    net = Network("three_bus_dc")
    net.add_bus(Bus(id="source", v_fixed=1.2))
    net.add_bus(Bus(id="busA", C=0.05, v_init=1.1, v_min=0.3))
    net.add_bus(Bus(id="busB", C=0.03, v_init=1.05, v_min=0.3))
    net.add_line(Line(id="line_src_A", from_bus="source", to_bus="busA", R=0.10, L=0.003))
    net.add_line(Line(id="line_A_B", from_bus="busA", to_bus="busB", R=0.08, L=0.002))
    net.add_component(ConstantPowerLoad(id="cpl_A", bus="busA", P=0.3))
    net.add_component(ConstantImpedanceLoad(id="zload_B", bus="busB", R=2.0))

    assembled = AutomaticModelBuilder.build(net)
    assert assembled.n_states == 4
    assert set(assembled.state_names) == {"v_busA", "v_busB", "i_line_src_A", "i_line_A_B"}

    eq = assembled.find_equilibrium(assembled.initial_guess(), u=assembled.default_input())
    assert eq.converged
    assert eq.is_stable

    sim = Simulator(assembled)
    disturbed = eq.x_star + np.array([0.1, -0.05, 0.02, 0.01])
    traj = sim.simulate(disturbed, (0.0, 5.0), u=assembled.default_input(), n_eval=100)
    assert traj.success
    assert np.linalg.norm(traj.final_state - eq.x_star) < 1e-3


def test_ideal_source_component():
    net = Network("with_ideal_source_component")
    net.add_bus(Bus(id="src_bus", C=0.02, v_init=1.2, v_min=0.1))
    net.add_bus(Bus(id="load_bus", C=0.02, v_init=1.15, v_min=0.1))
    net.add_component(IdealSource(id="src", bus="src_bus", v_source=1.2, R_source=0.02))
    net.add_line(Line(id="l1", from_bus="src_bus", to_bus="load_bus", R=0.1, L=0.002))
    net.add_component(ConstantImpedanceLoad(id="zl", bus="load_bus", R=5.0))

    assembled = AutomaticModelBuilder.build(net)
    eq = assembled.find_equilibrium(assembled.initial_guess(), u=assembled.default_input())
    assert eq.converged


if __name__ == "__main__":
    import sys
    import inspect
    fns = [f for name, f in inspect.getmembers(sys.modules[__name__], inspect.isfunction)
           if name.startswith("test_")]
    passed, failed = 0, 0
    for f in fns:
        try:
            f()
            print(f"PASS {f.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL {f.__name__} -> {e!r}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
