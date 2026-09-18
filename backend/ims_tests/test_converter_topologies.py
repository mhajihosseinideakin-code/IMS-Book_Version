"""
Tests for the ConverterModel base class and the Buck/Boost/BuckBoost
topology implementations (Phase V2.4).

The central tests here validate equilibrium against textbook,
independently-known-correct ideal (R_L=0) CCM steady-state voltage
conversion ratios -- the same "check against ground truth, not just
plausibility" discipline used for MRC synthesis (exact symbolic equality
against a published result) and the network assembler (exact numeric
equality against a hand-built model).
"""
import numpy as np

from ims_platform.network import (
    Network, Bus, ConstantImpedanceLoad, Converter,
    ElectricalModel, ConverterModel, BuckModel, BoostModel, BuckBoostModel, ConstantDutyController,
    AutomaticModelBuilder,
)
from ims_platform.core import Simulator


def _build_and_solve(em_cls, v_in, D, R_L, R_load=5.0, C=0.02, L=0.001, v_init=None):
    net = Network("conv_test")
    net.add_bus(Bus(id="out", C=C, v_init=v_init or v_in, v_min=0.05))
    em = em_cls(v_in=v_in, L=L, R_L=R_L)
    ctrl = ConstantDutyController(d=D)
    net.add_component(Converter(id="conv", bus="out", electrical_model=em, controller=ctrl))
    net.add_component(ConstantImpedanceLoad(id="load", bus="out", R=R_load))
    assembled = AutomaticModelBuilder.build(net)
    eq = assembled.find_equilibrium(assembled.initial_guess(), u=assembled.default_input())
    v_o = eq.x_star[assembled.state_names.index("v_out")]
    return assembled, eq, v_o


def test_converter_model_is_subclass_of_electrical_model():
    assert issubclass(ConverterModel, ElectricalModel)
    for cls in (BuckModel, BoostModel, BuckBoostModel):
        assert issubclass(cls, ConverterModel)


def test_clamp_duty_keeps_duty_away_from_degenerate_limits():
    assert ConverterModel.clamp_duty(-1.0) == 0.02
    assert ConverterModel.clamp_duty(2.0) == 0.98
    assert ConverterModel.clamp_duty(0.5) == 0.5


def test_buck_matches_textbook_ratio_ideal():
    V_IN = 12.0
    for D in (0.15, 0.3, 0.5, 0.7, 0.85):
        _, eq, v_o = _build_and_solve(BuckModel, V_IN, D, R_L=1e-6, v_init=D * V_IN)
        expected = D * V_IN
        assert eq.converged and eq.is_stable
        assert abs(v_o - expected) < 1e-4, f"D={D}: got {v_o}, expected {expected}"


def test_boost_matches_textbook_ratio_ideal():
    V_IN = 12.0
    for D in (0.15, 0.3, 0.5, 0.7, 0.85):
        _, eq, v_o = _build_and_solve(BoostModel, V_IN, D, R_L=1e-6, v_init=V_IN / (1 - D))
        expected = V_IN / (1 - D)
        assert eq.converged and eq.is_stable
        # Relative tolerance: R_L's effect on accuracy is amplified by the
        # (1-D) factor at high duty ratio (a real, expected property of
        # boost/buck-boost converters, not a numerical defect -- absolute
        # error grows with the output magnitude itself).
        assert abs(v_o - expected) / expected < 1e-3, f"D={D}: got {v_o}, expected {expected}"


def test_buckboost_matches_textbook_ratio_ideal():
    V_IN = 12.0
    for D in (0.15, 0.3, 0.5, 0.7, 0.85):
        _, eq, v_o = _build_and_solve(BuckBoostModel, V_IN, D, R_L=1e-6, v_init=D * V_IN / (1 - D))
        expected = D * V_IN / (1 - D)
        assert eq.converged and eq.is_stable
        assert abs(v_o - expected) / expected < 1e-3, f"D={D}: got {v_o}, expected {expected}"


def test_nonideal_converter_still_converges_to_stable_lower_voltage():
    """With nonzero R_L, output should still converge, to a slightly lower voltage than the ideal ratio."""
    V_IN = 12.0
    D = 0.4
    _, eq, v_o = _build_and_solve(BuckModel, V_IN, D, R_L=0.05, v_init=D * V_IN)
    assert eq.converged and eq.is_stable
    assert v_o < D * V_IN  # resistive drop reduces output below the ideal ratio
    assert v_o > 0.5 * D * V_IN  # but not collapsed


def test_output_voltage_independent_of_load_for_ideal_converter():
    """An ideal (R_L=0) converter's output voltage is set by duty ratio alone, independent of load resistance."""
    V_IN, D = 12.0, 0.4
    _, _, v_o_5ohm = _build_and_solve(BuckModel, V_IN, D, R_L=1e-6, R_load=5.0, v_init=D * V_IN)
    _, _, v_o_20ohm = _build_and_solve(BuckModel, V_IN, D, R_L=1e-6, R_load=20.0, v_init=D * V_IN)
    assert abs(v_o_5ohm - v_o_20ohm) < 1e-4


def test_full_pipeline_and_disturbance_recovery():
    assembled, eq, _ = _build_and_solve(BuckModel, 12.0, 0.4, R_L=0.05, v_init=0.4 * 12.0)
    sim = Simulator(assembled, method="RK45")
    disturbed = eq.x_star + np.array([0.2, -0.3])
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
