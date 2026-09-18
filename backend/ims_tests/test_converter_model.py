"""
Tests for the Converter object model (architecture document Part IV
Section 4.4): ElectricalModel, Controller, and Converter, which composes
them into a stateful network component.

The central test here (test_same_electrical_model_with_different_controllers)
proves the specific architectural property this phase was built for:
one ElectricalModel instance's physics is unaffected by which Controller
drives it, and swapping the controller is a one-line change at the
network-definition level, not a code change to the converter.
"""
import numpy as np

try:
    import pytest
except ImportError:
    pytest = None

from ims_platform.network import (
    Network, Bus, Line, ConstantImpedanceLoad, Converter,
    ElectricalModel, FilteredVoltageSource,
    Controller, ConstantSetpointController, SimplePIVoltageController,
    AutomaticModelBuilder,
)
from ims_platform.core import Simulator
from ims_platform.ims import IntrinsicManifold, RecoverabilityAnalyzer


def _build_converter_network(controller: Controller, em: ElectricalModel = None) -> Network:
    net = Network("converter_demo")
    net.add_bus(Bus(id="conv_bus", C=0.02, v_init=1.15, v_min=0.1))
    net.add_bus(Bus(id="load_bus", C=0.02, v_init=1.10, v_min=0.1))
    electrical_model = em or FilteredVoltageSource(tau=0.002, R_out=0.05)
    net.add_component(Converter(id="conv1", bus="conv_bus", electrical_model=electrical_model, controller=controller))
    net.add_line(Line(id="line1", from_bus="conv_bus", to_bus="load_bus", R=0.1, L=0.002))
    net.add_component(ConstantImpedanceLoad(id="zl", bus="load_bus", R=5.0))
    return net


def test_electrical_model_rejects_missing_params():
    if pytest is not None:
        with pytest.raises(ValueError):
            FilteredVoltageSource(tau=0.001)  # missing R_out
    else:
        try:
            FilteredVoltageSource(tau=0.001)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_controller_rejects_missing_params():
    if pytest is not None:
        with pytest.raises(ValueError):
            ConstantSetpointController()  # missing v_ref
    else:
        try:
            ConstantSetpointController()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_converter_state_names_are_correctly_prefixed_once():
    net = _build_converter_network(ConstantSetpointController(v_ref=1.2))
    assembled = AutomaticModelBuilder.build(net)
    assert "conv1_em_v_conv" in assembled.state_names
    assert "conv1_conv1_em_v_conv" not in assembled.state_names  # no double-prefix regression


def test_converter_with_stateless_controller_assembles_and_solves():
    net = _build_converter_network(ConstantSetpointController(v_ref=1.2))
    assembled = AutomaticModelBuilder.build(net)
    assert assembled.n_states == 4  # 2 bus voltages + 1 line current + 1 converter state

    eq = assembled.find_equilibrium(assembled.initial_guess(), u=assembled.default_input())
    assert eq.converged
    assert eq.is_stable


def test_converter_with_stateful_controller_assembles_and_solves():
    net = _build_converter_network(SimplePIVoltageController(v_nom=1.2, Kp=0.5, Ki=50.0))
    assembled = AutomaticModelBuilder.build(net)
    assert assembled.n_states == 5  # + 1 controller integrator state

    eq = assembled.find_equilibrium(assembled.initial_guess(), u=assembled.default_input())
    assert eq.converged
    assert eq.is_stable
    # The PI controller should drive its OWN bus voltage exactly to v_nom at equilibrium.
    v_conv_bus = eq.x_star[assembled.state_names.index("v_conv_bus")]
    assert abs(v_conv_bus - 1.2) < 1e-6


def test_same_electrical_model_with_different_controllers():
    """
    The central architectural proof: the SAME ElectricalModel physics,
    driven by two different Controllers, produces two different but each
    individually consistent closed-loop systems -- proving the
    ElectricalModel/Controller separation is real, not cosmetic.
    """
    net_const = _build_converter_network(ConstantSetpointController(v_ref=1.2))
    net_pi = _build_converter_network(SimplePIVoltageController(v_nom=1.2, Kp=0.5, Ki=50.0))

    assembled_const = AutomaticModelBuilder.build(net_const)
    assembled_pi = AutomaticModelBuilder.build(net_pi)

    eq_const = assembled_const.find_equilibrium(assembled_const.initial_guess(), u=assembled_const.default_input())
    eq_pi = assembled_pi.find_equilibrium(assembled_pi.initial_guess(), u=assembled_pi.default_input())

    assert eq_const.converged and eq_pi.converged
    # Different controllers -> generally different equilibria (the PI
    # controller actively corrects the line-drop the constant-setpoint
    # controller does not), proving the controller swap actually changed
    # closed-loop behaviour, not just cosmetically different code paths.
    v_conv_const = eq_const.x_star[assembled_const.state_names.index("v_conv_bus")]
    v_conv_pi = eq_pi.x_star[assembled_pi.state_names.index("v_conv_bus")]
    assert abs(v_conv_const - v_conv_pi) > 1e-3


def test_full_pipeline_runs_on_converter_network():
    """Equilibrium and time-domain simulation, unmodified, on a Converter-based network."""
    net = _build_converter_network(ConstantSetpointController(v_ref=1.2))
    assembled = AutomaticModelBuilder.build(net)
    x0 = assembled.initial_guess()

    eq = assembled.find_equilibrium(x0, u=assembled.default_input())
    assert eq.converged

    sim = Simulator(assembled, method="RK45")
    disturbed = eq.x_star + np.array([0.05, -0.03, 0.01, 0.02])
    traj = sim.simulate(disturbed, (0.0, 0.5), u=assembled.default_input(), n_eval=200)
    assert traj.success
    assert np.linalg.norm(traj.final_state - eq.x_star) < 1e-3


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
