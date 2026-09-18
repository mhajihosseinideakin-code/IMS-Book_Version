"""
Tests for the finalized controller interface: `SynthesizedMRCController`,
which adapts a control law derived by `control.mrc_synthesis.MRCSynthesizer`
(the IMS Control Framework, Phase V2.1) into a `network.controller.Controller`
that can drive a `network.converter.Converter` inside an assembled
Network (Phase V2.2-V2.4). This is the piece that proves the platform's
two most scientifically important capabilities -- automatic MRC
derivation, and automatic network assembly -- actually compose, not just
coexist.
"""
import numpy as np

from ims_platform.models import ConverterCPLPaper
from ims_platform.control import MRCSynthesizer
from ims_platform.network import (
    Network, Bus, ConstantPowerLoad, Converter, ConverterCPLElectricalModel,
    ConstantSetpointController, SynthesizedMRCController, AutomaticModelBuilder,
)
from ims_platform.core import Simulator

P_VALUES = dict(R=0.20, L=1.5e-3, C=2.5e-3, P=10e3)
KM_VALUE = 500.0


def _build_mrc_network():
    paper_model = ConverterCPLPaper()
    synth = MRCSynthesizer(paper_model)
    result = synth.synthesize()
    mrc_ctrl = SynthesizedMRCController(result, P_VALUES, KM_VALUE, state_order=[0, "bus", 1])

    net = Network("converter_cpl_network")
    net.add_bus(Bus(id="bus", C=P_VALUES["C"], v_init=400.0, v_min=40.0))
    em = ConverterCPLElectricalModel(R=P_VALUES["R"], L=P_VALUES["L"])
    net.add_component(Converter(id="conv", bus="bus", electrical_model=em, controller=mrc_ctrl))
    net.add_component(ConstantPowerLoad(id="cpl", bus="bus", P=P_VALUES["P"], v_floor=1e-6))
    return net, result, paper_model


def test_control_signal_via_network_adapter_matches_direct_synthesis_exactly():
    """The adapter must not alter the derived law's value at all -- it only reshapes arguments."""
    net, result, _ = _build_mrc_network()
    kappa_direct = result.compile_numeric(P_VALUES, KM_VALUE)
    mrc_ctrl = net.components[0].controller

    rng = np.random.default_rng(1)
    for _ in range(50):
        i_l, v_b, v_o = rng.uniform(-50, 100), rng.uniform(100, 600), rng.uniform(100, 600)
        u_direct = kappa_direct(np.array([i_l, v_b, v_o]), np.array([]))
        u_via_adapter = mrc_ctrl.control_signal(v_b, np.array([i_l, v_o]), np.array([]), None, {})
        assert abs(u_direct - u_via_adapter) < 1e-12


def test_assembled_network_open_loop_physics_matches_standalone_paper_model():
    """
    The network decomposition's open-loop (u=0) physics must match the
    standalone ConverterCPLPaper model, up to the small, quantitatively
    understood difference introduced by ConstantPowerLoad's smooth
    voltage-floor clamp (which the standalone model does not use) --
    O(1e-7) relative to the dynamics magnitude at realistic voltages,
    scaling as 1/v^2 and bounded even near the low end of the tested range.
    """
    net, _, paper_model = _build_mrc_network()
    assembled = AutomaticModelBuilder.build(net)

    rng = np.random.default_rng(0)
    for _ in range(100):
        i_l, v_b, v_o = rng.uniform(-50, 100), rng.uniform(100, 600), rng.uniform(100, 600)
        f_standalone = paper_model.dynamics(0.0, np.array([i_l, v_b, v_o]), np.array([0.0]), paper_model.params)
        x_assembled = np.array([v_b, i_l, v_o])
        f_assembled = assembled.dynamics(0.0, x_assembled, np.zeros(0), assembled.params)
        # assembled order [dv_b, di_l, dv_o]; standalone order [di_l, dv_b, dv_o]
        rel_diff_di = abs(f_assembled[1] - f_standalone[0]) / max(1.0, abs(f_standalone[0]))
        rel_diff_dv = abs(f_assembled[0] - f_standalone[1]) / max(1.0, abs(f_standalone[1]))
        assert rel_diff_di < 1e-8
        assert rel_diff_dv < 1e-6


def test_synthesized_mrc_controller_drives_recovery_in_assembled_network():
    """
    The end-to-end proof: a control law derived entirely automatically
    (Symbolic Engine -> MRCSynthesizer), attached to a Converter through
    the finalized controller interface, recovers the network from a
    disturbance -- the manifold residual contracts at exactly rate k_m,
    the same guarantee already proven for the standalone model, now also
    holding through the network/assembler layer.
    """
    net, _, _ = _build_mrc_network()
    assembled = AutomaticModelBuilder.build(net)

    x_star = np.array([400.0, 25.0, 405.0])  # v_bus, i_l, v_o -- known equilibrium
    x0 = x_star + np.array([-5.0, 0.0, 0.0])  # 5V bus deficit, matching the paper's disturbance study

    sim = Simulator(assembled, method="RK45")
    traj = sim.simulate(x0, (0.0, 0.02), u=np.zeros(0), n_eval=400)
    assert traj.success

    def residual(x):
        v_bus, i_l, v_o = x
        return v_bus - (v_o - P_VALUES["R"] * i_l)

    em0 = residual(x0)
    idx = np.argmin(np.abs(traj.t - 0.01))
    predicted = em0 * np.exp(-KM_VALUE * traj.t[idx])
    actual = residual(traj.x[:, idx])
    assert abs(actual - predicted) < 0.05 * abs(em0)


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
