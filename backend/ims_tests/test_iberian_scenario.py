"""
Tests for server.iberian_scenario -- the "Iberian 2025-Inspired
Overvoltage Cascade" demo case (funding-demo brief).

The central claim this file locks in place: at the frozen
DEFAULT_NETWORK_PARAMS, running the exact same scripted three-event
disturbance with only the KQ (dynamic voltage support) slider changed
produces two genuinely different, independently-verified outcomes --

    KQ=0 (fixed power factor)  -> settles to a stable equilibrium ABOVE
                                   the overvoltage-trip threshold,
                                   protection stuck engaged: NON-RECOVERABLE
    KQ=6 (dynamic Q support)   -> settles to a stable equilibrium BELOW
                                   the threshold, protection clear: RECOVERABLE

and that both endpoints are genuine small-signal-stable equilibria (not
one stable + one merely-not-yet-diverged, and not a scripted/assumed
outcome) -- the textbook IMS point that small-signal stability and
large-signal recoverability are distinct questions.

If any of these tests start failing after a change to
network/renewable_components.py or server/iberian_scenario.py's
DEFAULT_NETWORK_PARAMS, that change has broken the scenario's core
physical story and must not be shipped without re-establishing it
(re-run find the parameter sweep / re-derive KQ_RECOVERABLE_REFERENCE)
before anything UI-facing is touched.
"""
import numpy as np

try:
    import pytest
except ImportError:
    pytest = None

from ims_platform.server import iberian_scenario as isc


def test_nonrecoverable_reference_settles_above_trip_and_stable():
    r = isc.run_stress_test(
        {"KQ": isc.KQ_NONRECOVERABLE_REFERENCE, "Q_avail": isc.SLIDER_DEFAULTS["Q_avail"],
         "Imax": isc.SLIDER_DEFAULTS["Imax"]},
        severity=1.0,
    )
    assert r["baseline_ok"]
    assert r["baseline"]["small_signal_stable"]  # pre-disturbance point is conventionally "stable"
    s = r["summary"]
    assert s["verdict"] == "NON-RECOVERABLE"
    assert s["final_v_es_sw"] > r["v_trip"], "must genuinely exceed the trip threshold, not just approach it"
    assert s["final_trip_timer"] > 0.9, "protection must be persistently, not just transiently, engaged"
    assert s["min_voltage_margin"] < 0


def test_recoverable_reference_settles_below_trip_and_stable():
    r = isc.run_stress_test(
        {"KQ": isc.KQ_RECOVERABLE_REFERENCE, "Q_avail": isc.SLIDER_DEFAULTS["Q_avail"],
         "Imax": isc.SLIDER_DEFAULTS["Imax"]},
        severity=1.0,
    )
    assert r["baseline_ok"]
    assert r["baseline"]["small_signal_stable"]
    s = r["summary"]
    assert s["verdict"] == "RECOVERABLE"
    assert s["final_v_es_sw"] < r["v_trip"]
    assert s["final_trip_timer"] < 0.2, "protection must genuinely clear, not just improve"
    assert s["min_voltage_margin"] > 0


def test_final_states_are_independently_verified_stable_equilibria():
    """
    Cross-check using the platform's own generic, scenario-agnostic
    Jacobian/eigenvalue machinery (System.find_equilibrium), independent
    of this scenario's own verdict logic: both endpoints must be genuine
    equilibria (near-zero residual, all eigenvalues in the open left
    half-plane) reached via full time-domain simulation, not asserted.
    """
    for KQ, expect_over_trip in [(isc.KQ_NONRECOVERABLE_REFERENCE, True),
                                  (isc.KQ_RECOVERABLE_REFERENCE, False)]:
        system, hist_components, used = isc._build_system(
            {"KQ": KQ, "Q_avail": isc.SLIDER_DEFAULTS["Q_avail"], "Imax": isc.SLIDER_DEFAULTS["Imax"]}
        )
        x0 = system.initial_guess()
        eq0 = system.find_equilibrium(x0, u=np.array([used["P_fleet"]]))
        assert eq0.converged and eq0.is_stable

        t, x, ok = isc._simulate_cascade(system, hist_components, used["P_fleet"], eq0.x_star,
                                          severity=1.0, tail_horizon=6.0)
        assert ok
        final_state = x[:, -1]
        eq_final = system.find_equilibrium(final_state, u=np.array([used["P_fleet"]]))

        assert eq_final.converged
        assert eq_final.is_stable, f"KQ={KQ}: post-disturbance settling point must be a genuine stable equilibrium"
        assert np.linalg.norm(final_state - eq_final.x_star) < 1e-2, "trajectory must have actually settled onto it"

        v_sw = final_state[system.state_names.index("v_es_sw")]
        v_trip = isc.DEFAULT_NETWORK_PARAMS["v_trip"]
        if expect_over_trip:
            assert v_sw > v_trip
        else:
            assert v_sw < v_trip


def test_kq_sweep_is_monotonic_between_the_two_reference_points():
    """The bifurcation is a genuine, monotonic function of KQ, not a fluke at the two chosen reference values."""
    kqs = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0]
    finals = []
    for kq in kqs:
        r = isc.run_stress_test({"KQ": kq, "Q_avail": isc.SLIDER_DEFAULTS["Q_avail"],
                                  "Imax": isc.SLIDER_DEFAULTS["Imax"]}, severity=1.0)
        assert r["baseline_ok"]
        finals.append(r["summary"]["final_v_es_sw"])
    diffs = np.diff(finals)
    assert np.all(diffs <= 1e-6), f"final v_es_sw should be non-increasing in KQ, got {finals}"
    # and it must actually cross the trip threshold somewhere in this range
    v_trip = isc.DEFAULT_NETWORK_PARAMS["v_trip"]
    assert finals[0] > v_trip > finals[-1]


def test_verify_baseline_bifurcation_helper_passes():
    """The module's own executable self-check (imported/used by the server on demand) must pass standalone."""
    result = isc.verify_baseline_bifurcation()
    assert result["non_recoverable_case"]["summary"]["verdict"] == "NON-RECOVERABLE"
    assert result["recoverable_case"]["summary"]["verdict"] == "RECOVERABLE"


def test_disclaimer_present_in_every_public_response():
    for resp in [
        isc.build_summary({}),
        isc.run_stress_test({}),
        isc.run_boundary_map({}, severities=[1.0]),
    ]:
        assert resp.get("disclaimer"), "every public response must carry the illustrative-model disclaimer"
        assert "NOT a reconstruction" in resp["disclaimer"]
