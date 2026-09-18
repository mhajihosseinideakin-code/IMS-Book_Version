"""
Basic correctness tests for the IMS Platform core engine.
Run with: pytest tests/
"""
import numpy as np

try:
    import pytest
except ImportError:  # pragma: no cover - allows running without pytest installed
    pytest = None

from ims_platform.models import GridFormingInverter, DCMicrogridCPL
from ims_platform.core import Simulator
from ims_platform.ims import IntrinsicManifold, RecoverabilityAnalyzer
from ims_platform.control import ManifoldReshapingControl


def test_inverter_equilibrium_is_stable_near_nominal():
    sys_model = GridFormingInverter()
    eq = sys_model.find_equilibrium(np.array([0.3, 0.0]), u=np.array([0.5]))
    assert eq.converged
    assert eq.is_stable
    assert eq.residual_norm < 1e-6


def test_inverter_manifold_traces_expected_number_of_points():
    sys_model = GridFormingInverter()
    M = IntrinsicManifold(sys_model, param_name="P_set").build(
        alpha_range=(0.0, 0.9), n_points=40, x0_guess=np.array([0.3, 0.0])
    )
    assert len(M.points) == 40
    assert all(p.is_stable for p in M.points)


def test_manifold_residual_zero_on_manifold():
    sys_model = GridFormingInverter()
    M = IntrinsicManifold(sys_model, param_name="P_set").build(
        alpha_range=(0.0, 0.9), n_points=40, x0_guess=np.array([0.3, 0.0])
    )
    on_manifold_point = M.points[10].x_star
    assert M.residual(on_manifold_point) < 1e-8


def test_manifold_residual_positive_off_manifold():
    sys_model = GridFormingInverter()
    M = IntrinsicManifold(sys_model, param_name="P_set").build(
        alpha_range=(0.0, 0.9), n_points=40, x0_guess=np.array([0.3, 0.0])
    )
    off_point = M.points[10].x_star + np.array([1.0, 1.0])
    assert M.residual(off_point) > 0.5


def test_simulator_open_loop_converges_to_equilibrium():
    sys_model = GridFormingInverter()
    sim = Simulator(sys_model)
    eq = sys_model.find_equilibrium(np.array([0.3, 0.0]), u=np.array([0.5]))
    traj = sim.simulate(eq.x_star + np.array([0.3, 0.5]), (0.0, 15.0), u=np.array([0.5]), n_eval=300)
    assert traj.success
    assert np.linalg.norm(traj.final_state - eq.x_star) < 1e-2


def test_dc_microgrid_has_stable_and_saddle_equilibria():
    sys_model = DCMicrogridCPL()
    analytic = sys_model.equilibria_analytic(0.5)
    assert analytic.shape[0] == 2
    stable_found, saddle_found = False, False
    for i_star, v_star in analytic:
        eq = sys_model.find_equilibrium(np.array([i_star, v_star]), u=np.array([0.5]))
        assert eq.converged
        if eq.is_stable:
            stable_found = True
        elif np.any(eq.eigenvalues.imag == 0) and np.any(eq.eigenvalues.real > 0) and np.any(eq.eigenvalues.real < 0):
            saddle_found = True
    assert stable_found
    assert saddle_found


def test_recoverability_report_fields_consistent():
    sys_model = GridFormingInverter()
    sim = Simulator(sys_model)
    M = IntrinsicManifold(sys_model, param_name="P_set").build(
        alpha_range=(0.0, 0.9), n_points=40, x0_guess=np.array([0.3, 0.0])
    )
    eq = sys_model.find_equilibrium(np.array([0.3, 0.0]), u=np.array([0.5]))
    analyzer = RecoverabilityAnalyzer(sys_model, M, sim)
    report = analyzer.assess(eq.x_star, radius=1.0, n_samples=25, t_horizon=8.0, u=np.array([0.5]))
    assert report.n_samples == 25
    assert report.n_recoverable == int(np.sum(report.labels))
    assert 0.0 <= report.recoverability_index <= 1.0
    assert report.risk_level in ("LOW", "MEDIUM", "HIGH")


def test_mrc_gain_schedule_matches_manifold_length():
    sys_model = GridFormingInverter()
    M = IntrinsicManifold(sys_model, param_name="P_set").build(
        alpha_range=(0.0, 0.9), n_points=30, x0_guess=np.array([0.3, 0.0])
    )
    mrc = ManifoldReshapingControl(sys_model, M)
    mrc.precompute_gain_schedule(u_nominal=np.array([0.5]))
    assert len(mrc._gain_schedule) == len(M.points)


def test_mrc_converges_close_to_manifold_after_settling():
    """
    MRC should settle the closed-loop trajectory back onto (near) the
    intrinsic manifold after a large disturbance, once transients have
    died out. (We check convergence at a fully-settled horizon rather
    than mid-transient, since a proportional local-LQR law can have a
    temporarily larger residual than open loop during the transient
    itself even though it is regulating toward a moving local target.)
    """
    sys_model = GridFormingInverter()
    sim = Simulator(sys_model)
    M = IntrinsicManifold(sys_model, param_name="P_set").build(
        alpha_range=(0.0, 0.9), n_points=40, x0_guess=np.array([0.3, 0.0])
    )
    eq = sys_model.find_equilibrium(np.array([0.3, 0.0]), u=np.array([0.5]))
    disturbed = eq.x_star + np.array([1.0, 2.0])

    mrc = ManifoldReshapingControl(sys_model, M, Q=np.diag([8.0, 2.0]), R=np.array([[0.05]]))
    controller = mrc.as_controller(u_nominal=np.array([0.5]))
    traj_cl = sim.simulate(disturbed, (0.0, 15.0), controller=controller, n_eval=300)

    assert traj_cl.success
    assert M.residual(traj_cl.final_state) < 0.05


def test_mrc_recovers_a_case_open_loop_cannot():
    """
    On the bistable DC-microgrid model, find a disturbed state that
    diverges (voltage collapse) in open loop, and confirm MRC recovers
    it -- i.e. MRC genuinely enlarges the recoverable region rather than
    just re-deriving the open-loop result.
    """
    sys_model = DCMicrogridCPL()
    sim = Simulator(sys_model, method="RK45")
    M = IntrinsicManifold(sys_model, param_name="P_load").build(
        alpha_range=(0.05, 1.0), n_points=40, x0_guess=np.array([0.1, 1.19]), keep_unstable=False
    )
    nominal_u = np.array([0.5])
    eq = sys_model.find_equilibrium(np.array([0.44, 1.13]), u=nominal_u)

    # A disturbed state on the low-voltage side of the saddle: verified to
    # diverge (voltage collapse) in open loop.
    disturbed = np.array([0.68894065, 0.29546767])

    traj_ol = sim.simulate(disturbed, (0.0, 3.0), u=nominal_u, n_eval=200)
    ol_recovered = traj_ol.success and sys_model.admissible(traj_ol.final_state) and M.residual(traj_ol.final_state) < 0.1

    mrc = ManifoldReshapingControl(
        sys_model, M, Q=np.diag([1.0, 25.0]), R=np.array([[0.5]]),
        u_min=np.array([0.05]), u_max=np.array([1.5]),
    )
    controller = mrc.as_controller(u_nominal=nominal_u)
    traj_cl = sim.simulate(disturbed, (0.0, 3.0), controller=controller, n_eval=200)
    cl_recovered = traj_cl.success and sys_model.admissible(traj_cl.final_state) and M.residual(traj_cl.final_state) < 0.1

    assert not ol_recovered, "expected this disturbance to be non-recoverable in open loop for the test to be meaningful"
    assert cl_recovered, "MRC should recover a disturbance that diverges in open loop"


if __name__ == "__main__":
    import sys
    if pytest is not None:
        sys.exit(pytest.main([__file__, "-v"]))
    else:
        # Minimal standalone runner if pytest isn't installed in this environment.
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
