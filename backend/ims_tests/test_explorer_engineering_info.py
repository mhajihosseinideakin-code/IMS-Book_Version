"""
Tests for the backend enhancements addressing the IMS Platform Explorer
v1.0 technical review's Priority 1 (Network Information Panel) and
Priority 2 (Visualization / rich engineering results) items:

    - network_summary / component_chain, computed from the REAL
      assembled Network object for converter_topology/network_mrc
      projects (not hand-maintained descriptions that could drift).
    - solver_diagnostics on the equilibrium stage (function evaluations,
      Jacobian condition number, computation time) -- exposing
      information the solver already computes internally.
    - performance_metrics (max deviation, steady-state error, settling
      time, overshoot) at the simulate stage, validated here against a
      synthetic case with analytically-known behaviour, not just
      "did it return a number".
    - manifold_residual_trajectory at the IMS Analysis stage for
      single_model/converter_topology, matching what network_mrc
      already provided.
"""
import numpy as np

from ims_platform.server import create_app
from ims_platform.server.app import _performance_metrics


def _client():
    return create_app().test_client()


def test_performance_metrics_on_synthetic_damped_oscillation():
    """
    x(t) = x_star + A*exp(-t)*cos(3t): a case whose behaviour is known
    well enough to sanity-check the metric definitions themselves,
    independent of any real system's dynamics.
    """
    t = np.linspace(0, 8, 2000)
    x_star = np.array([2.0])
    A = 1.0
    x = (x_star[0] + A * np.exp(-t) * np.cos(3 * t))[None, :]

    m = _performance_metrics(t, x, x_star)
    assert abs(m["max_deviation"][0] - A) < 0.05      # peak is at t=0 for this signal
    assert m["steady_state_error"][0] < 0.01           # decayed to ~0 by t=8
    assert m["overshoot_pct"][0] > 0                   # a damped cosine does overshoot
    assert m["settling_time"][0] is not None
    assert 0 < m["settling_time"][0] < 8


def test_performance_metrics_zero_initial_deviation_gives_zero_overshoot():
    t = np.linspace(0, 5, 100)
    x_star = np.array([1.0])
    x = np.full((1, 100), 1.0)  # already at equilibrium throughout
    m = _performance_metrics(t, x, x_star)
    assert m["overshoot_pct"][0] == 0.0
    assert m["max_deviation"][0] == 0.0
    assert m["settling_time"][0] == 0.0


def test_performance_metrics_never_settling_returns_none():
    t = np.linspace(0, 5, 100)
    x_star = np.array([0.0])
    x = np.ones((1, 100)) * 10.0  # constant, large deviation, never approaches x_star
    m = _performance_metrics(t, x, x_star)
    assert m["settling_time"][0] is None


def test_build_stage_network_summary_matches_real_network_for_converter_topology():
    c = _client()
    r = c.post("/api/project/buck_converter/build", json={"params": {}, "nominal_input": [0.4]})
    d = r.get_json()
    ns = d["network_summary"]
    assert ns["buses"] == 1
    assert ns["converters"] == 1
    assert ns["loads"] == 1
    assert ns["controllers"] == 1
    assert len(d["component_chain"]) >= 2
    assert d["model_order"]["n_states"] == 2


def test_build_stage_network_summary_for_network_mrc():
    c = _client()
    r = c.post("/api/project/network_auto_mrc/build", json={})
    d = r.get_json()
    assert d["network_summary"]["converters"] == 1
    assert d["network_summary"]["loads"] == 1
    assert "IMS-Native MRC" in d["controller_description"]


def test_build_stage_network_summary_for_single_model():
    c = _client()
    r = c.post("/api/project/dc_microgrid_cpl/build", json={"params": {}, "nominal_input": [0.5]})
    d = r.get_json()
    assert d["network_summary"]["loads"] == 1
    assert len(d["component_chain"]) >= 2


def test_equilibrium_stage_includes_solver_diagnostics():
    c = _client()
    r = c.post("/api/project/grid_forming_inverter/equilibrium",
               json={"params": {}, "state_guess": [0.3, 0.0], "nominal_input": [0.5]})
    diag = r.get_json()["solver_diagnostics"]
    assert diag["function_evaluations"] > 0
    assert diag["jacobian_condition_number"] > 0
    assert diag["computation_time_s"] >= 0


def test_simulate_stage_includes_performance_metrics():
    c = _client()
    r = c.post("/api/project/grid_forming_inverter/equilibrium",
               json={"params": {}, "state_guess": [0.3, 0.0], "nominal_input": [0.5]})
    eq = r.get_json()
    r = c.post("/api/project/grid_forming_inverter/simulate", json={
        "params": {}, "nominal_input": [0.5], "x_star": eq["x_star"],
        "disturbance": {"type": "offset", "value": [1.0, 2.0]}, "horizon": 10.0,
    })
    d = r.get_json()
    metrics = d["performance_metrics_open_loop"]
    assert len(metrics["max_deviation"]) == 2
    assert len(metrics["settling_time"]) == 2
    # this specific disturbance is known-recoverable (Phase V1 validation); steady-state error should be tiny
    assert all(e < 0.01 for e in metrics["steady_state_error"])


def test_ims_analysis_stage_includes_manifold_residual_trajectory():
    c = _client()
    r = c.post("/api/project/grid_forming_inverter/equilibrium",
               json={"params": {}, "state_guess": [0.3, 0.0], "nominal_input": [0.5]})
    eq = r.get_json()
    payload = {
        "params": {}, "nominal_input": [0.5], "x_star": eq["x_star"],
        "disturbance": {"type": "offset", "value": [1.0, 2.0]}, "horizon": 10.0,
        "sweep_range": [0.0, 0.9], "sweep_points": 30,
        "radius": 1.5, "n_samples": 20, "recovery_tol": 0.15,
    }
    r = c.post("/api/project/grid_forming_inverter/ims_analysis", json=payload)
    d = r.get_json()
    mrt = d["manifold_residual_trajectory"]
    assert len(mrt["t"]) == len(mrt["residual"])
    assert mrt["residual"][0] > mrt["residual"][-1]  # residual should decay for a recoverable case


def test_network_mrc_simulate_includes_control_signal_and_metrics():
    c = _client()
    r = c.post("/api/project/network_auto_mrc/equilibrium", json={})
    eq = r.get_json()
    r = c.post("/api/project/network_auto_mrc/simulate", json={"x_star": eq["x_star"]})
    d = r.get_json()
    assert len(d["control_signal"]) == len(d["trajectory_open_loop"]["t"])
    assert "performance_metrics_open_loop" in d


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
