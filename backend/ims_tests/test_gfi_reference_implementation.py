"""
Tests for the Grid-Forming Inverter "reference implementation" pass --
per the new one-model-at-a-time development strategy, every addition
here is deliberately gated to this project specifically (checked here
by also confirming the OTHER projects are unaffected), and every
physical quantity reuses the model's own already-validated methods
(GridFormingInverter.electrical_power, .delta_max_frac_pi, .omega_max)
rather than a separately-maintained formula that could drift from the
actual dynamics.
"""
import numpy as np

from ims_platform.server import create_app
from ims_platform.models import GridFormingInverter


def _client():
    return create_app().test_client()


def test_admissible_bounds_are_introspectable_and_match_behavior():
    """The exposed bounds must be the SAME values admissible() actually enforces, not a separate copy."""
    m = GridFormingInverter()
    just_inside_delta = m.delta_max_frac_pi * np.pi - 0.01
    just_outside_delta = m.delta_max_frac_pi * np.pi + 0.01
    assert m.admissible(np.array([just_inside_delta, 0.0])) is True
    assert m.admissible(np.array([just_outside_delta, 0.0])) is False
    assert m.admissible(np.array([0.0, m.omega_max - 0.1])) is True
    assert m.admissible(np.array([0.0, m.omega_max + 0.1])) is False


def test_build_stage_includes_governing_equations_and_envelope_for_gfi_only():
    c = _client()
    r = c.post("/api/project/grid_forming_inverter/build", json={"params": {}, "nominal_input": [0.5]})
    d = r.get_json()
    assert len(d["governing_equations"]) == 3
    assert "P_e" in d["governing_equations"][2]
    assert d["admissible_envelope"]["omega_max"] == 25.0

    for pid, payload in [
        ("dc_microgrid_cpl", {"params": {}, "nominal_input": [0.5]}),
        ("buck_converter", {"params": {}, "nominal_input": [0.4]}),
        ("network_auto_mrc", {}),
    ]:
        r2 = c.post(f"/api/project/{pid}/build", json=payload)
        d2 = r2.get_json()
        assert "governing_equations" not in d2, f"{pid} should not have governing_equations"
        assert "admissible_envelope" not in d2, f"{pid} should not have admissible_envelope"


def test_equilibrium_electrical_power_matches_power_setpoint_exactly():
    """
    The defining validation for this addition: at equilibrium, the
    swing equation's own stationarity condition (-omega + mp*(P_set -
    P_e) = 0, omega = 0) forces P_e(delta*) = P_set exactly. Any
    computed electrical power that doesn't match P_set to solver
    tolerance indicates a bug in either the equilibrium solve or the
    power computation itself.
    """
    c = _client()
    for p_set in (0.2, 0.5, 0.8):
        r = c.post("/api/project/grid_forming_inverter/equilibrium",
                   json={"params": {}, "state_guess": [0.3, 0.0], "nominal_input": [p_set]})
        d = r.get_json()
        eop = d["electrical_operating_point"]
        assert abs(eop["electrical_power"] - p_set) < 1e-6
        assert eop["matches_setpoint"] is True

    r2 = c.post("/api/project/dc_microgrid_cpl/equilibrium",
               json={"params": {}, "state_guess": [0.44, 1.13], "nominal_input": [0.5]})
    assert "electrical_operating_point" not in r2.get_json()


def test_characteristic_time_constants_match_eigenvalues():
    c = _client()
    r = c.post("/api/project/grid_forming_inverter/equilibrium",
               json={"params": {}, "state_guess": [0.3, 0.0], "nominal_input": [0.5]})
    d = r.get_json()
    tau = d["characteristic_time_constants"]
    eigs = d["eigenvalues"]
    for t, e in zip(tau, eigs):
        assert abs(t - 1.0 / abs(e["re"])) < 1e-9


def test_electrical_power_trajectory_converges_to_setpoint():
    c = _client()
    eq = c.post("/api/project/grid_forming_inverter/equilibrium",
               json={"params": {}, "state_guess": [0.3, 0.0], "nominal_input": [0.5]}).get_json()
    r = c.post("/api/project/grid_forming_inverter/simulate", json={
        "params": {}, "nominal_input": [0.5], "x_star": eq["x_star"],
        "disturbance": {"type": "offset", "value": [1.0, 2.0]}, "horizon": 10.0,
    })
    d = r.get_json()
    p_e = d["electrical_power_open_loop"]["power"]
    assert abs(p_e[-1] - 0.5) < 0.01
    assert abs(p_e[0] - 0.5) > 0.5

    eq2 = c.post("/api/project/dc_microgrid_cpl/equilibrium",
               json={"params": {}, "state_guess": [0.44, 1.13], "nominal_input": [0.5]}).get_json()
    r2 = c.post("/api/project/dc_microgrid_cpl/simulate", json={
        "params": {}, "nominal_input": [0.5], "x_star": eq2["x_star"],
        "disturbance": {"type": "absolute", "value": [0.5, 1.0]}, "horizon": 2.0,
    })
    assert "electrical_power_open_loop" not in r2.get_json()


def test_electrical_power_closed_loop_present_when_mrc_enabled():
    c = _client()
    eq = c.post("/api/project/grid_forming_inverter/equilibrium",
               json={"params": {}, "state_guess": [0.3, 0.0], "nominal_input": [0.5]}).get_json()
    r = c.post("/api/project/grid_forming_inverter/simulate", json={
        "params": {}, "nominal_input": [0.5], "x_star": eq["x_star"],
        "disturbance": {"type": "offset", "value": [1.0, 2.0]}, "horizon": 10.0,
        "mrc_enabled": True, "Q_diag": [8, 2], "R_diag": [0.05], "u_min": [-0.2], "u_max": [1.2],
    })
    d = r.get_json()
    assert "electrical_power_closed_loop" in d
    # Scheduled LQR here has no integral action; under the specified
    # saturation limits (u_min=-0.2, u_max=1.2) it settles with a small
    # genuine steady-state offset (delta converges to ~0.1577 rather
    # than the true equilibrium's 0.1506) -- confirmed by checking the
    # state trajectory directly, not assumed. The tolerance below
    # reflects that real controller behaviour, not numerical noise.
    assert abs(d["electrical_power_closed_loop"]["power"][-1] - 0.5) < 0.05


def test_jacobian_matches_hand_derived_linearization():
    """
    Analytical cross-check independent of the platform's own Jacobian
    code: d(domega/dt)/ddelta = -(mp/tau_p)*(E*V/X)*cos(delta*), derived
    by hand from the swing equation, must match the returned matrix
    exactly (to solver precision).
    """
    c = _client()
    r = c.post("/api/project/grid_forming_inverter/equilibrium",
               json={"params": {}, "state_guess": [0.3, 0.0], "nominal_input": [0.5]})
    d = r.get_json()
    delta_star = d["x_star"][0]
    E, V, X, tau_p, m_p = 1.0, 1.0, 0.3, 0.05, 1.0
    dPe_ddelta = (E * V / X) * np.cos(delta_star)
    J10_expected = -(m_p / tau_p) * dPe_ddelta
    J11_expected = -1.0 / tau_p
    J = d["jacobian"]
    assert abs(J[0][0] - 0.0) < 1e-9
    assert abs(J[0][1] - 1.0) < 1e-9
    assert abs(J[1][0] - J10_expected) < 1e-6
    assert abs(J[1][1] - J11_expected) < 1e-9


def test_is_hyperbolic_and_transverse_stability_present():
    c = _client()
    r = c.post("/api/project/grid_forming_inverter/equilibrium",
               json={"params": {}, "state_guess": [0.3, 0.0], "nominal_input": [0.5]})
    d = r.get_json()
    assert d["is_hyperbolic"] is True
    assert d["transverse_stability"]["is_hyperbolic"] is True


def test_power_mismatch_is_zero_at_true_equilibrium():
    c = _client()
    r = c.post("/api/project/grid_forming_inverter/equilibrium",
               json={"params": {}, "state_guess": [0.3, 0.0], "nominal_input": [0.5]})
    d = r.get_json()
    assert abs(d["electrical_operating_point"]["power_mismatch"]) < 1e-6


def test_electrical_power_metrics_computed_and_gated_to_gfi():
    c = _client()
    eq = c.post("/api/project/grid_forming_inverter/equilibrium",
               json={"params": {}, "state_guess": [0.3, 0.0], "nominal_input": [0.5]}).get_json()
    r = c.post("/api/project/grid_forming_inverter/simulate", json={
        "params": {}, "nominal_input": [0.5], "x_star": eq["x_star"],
        "disturbance": {"type": "offset", "value": [1.0, 2.0]}, "horizon": 10.0,
    })
    d = r.get_json()
    m = d["electrical_power_metrics_open_loop"]
    assert m["steady_state_error"] < 0.01  # P_e settles back to P_set
    assert m["max_deviation"] > 1.0        # genuinely disturbed initially

    eq2 = c.post("/api/project/dc_microgrid_cpl/equilibrium",
               json={"params": {}, "state_guess": [0.44, 1.13], "nominal_input": [0.5]}).get_json()
    r2 = c.post("/api/project/dc_microgrid_cpl/simulate", json={
        "params": {}, "nominal_input": [0.5], "x_star": eq2["x_star"],
        "disturbance": {"type": "absolute", "value": [0.5, 1.0]}, "horizon": 2.0,
    })
    assert "electrical_power_metrics_open_loop" not in r2.get_json()


def test_jacobian_generically_present_for_other_projects_too():
    """
    Unlike the GFI-gated fields, the raw Jacobian is a generic,
    already-computed EquilibriumResult field with no new risk -- it is
    exposed for every project, not just grid_forming_inverter.
    """
    c = _client()
    r = c.post("/api/project/dc_microgrid_cpl/equilibrium",
               json={"params": {}, "state_guess": [0.44, 1.13], "nominal_input": [0.5]})
    d = r.get_json()
    assert d["jacobian"] is not None
    assert len(d["jacobian"]) == 2


def test_engineering_outputs_present_for_gfi_only():
    c = _client()
    r = c.post("/api/project/grid_forming_inverter/build", json={"params": {}, "nominal_input": [0.5]})
    d = r.get_json()
    names = [o["name"] for o in d["engineering_outputs"]]
    assert any("Electrical power" in n for n in names)

    r2 = c.post("/api/project/dc_microgrid_cpl/build", json={"params": {}, "nominal_input": [0.5]})
    assert "engineering_outputs" not in r2.get_json()


def test_recoverability_interpretation_covers_all_three_cases():
    from ims_platform.server.app import _recoverability_interpretation

    class FakeReport:
        def __init__(self, n, k, risk):
            self.n_samples, self.n_recoverable, self.risk_level = n, k, risk

    full = _recoverability_interpretation(FakeReport(20, 20, "LOW"))
    assert "fully recoverable" in full and "20" in full

    none = _recoverability_interpretation(FakeReport(20, 0, "HIGH"))
    assert "None" in none

    partial = _recoverability_interpretation(FakeReport(20, 12, "MEDIUM"))
    assert "12 of 20" in partial and "60%" in partial and "MEDIUM" in partial


def test_recoverability_interpretation_wired_into_real_response():
    c = _client()
    r = c.post("/api/project/grid_forming_inverter/equilibrium",
               json={"params": {}, "state_guess": [0.3, 0.0], "nominal_input": [0.5]})
    eq = r.get_json()
    payload = {
        "params": {}, "nominal_input": [0.5], "x_star": eq["x_star"],
        "sweep_range": [0.0, 0.9], "sweep_points": 30, "radius": 1.5, "n_samples": 20, "recovery_tol": 0.15,
    }
    r = c.post("/api/project/grid_forming_inverter/ims_analysis", json=payload)
    interp = r.get_json()["recoverability"]["interpretation"]
    assert "recoverable" in interp.lower()


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
