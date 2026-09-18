"""Tests for server.gfm_current_limit_case -- the "GFM Current-Limit Recovery" Case Library entry."""
import numpy as np

from ims_platform.server import gfm_current_limit_case as gcc


def test_reference_bifurcation():
    r = gcc.verify_reference_bifurcation()
    assert r["non_recoverable_case"]["summary"]["verdict"] == "NON-RECOVERABLE"
    assert r["recoverable_case"]["summary"]["verdict"] == "RECOVERABLE"


def test_every_fault_returns_to_same_prefault_equilibrium_regardless_of_imax():
    """This case's own documented claim: recoverability here is an FRT-depth criterion, not a
    bistable-equilibrium bifurcation -- every tested Imax must settle back to the pre-fault point."""
    for imax in [0.3, 0.5, 1.0, 2.0, 4.0]:
        r = gcc.run_stress_test({"Imax": imax})
        assert r["baseline_ok"]
        assert r["summary"]["settled_to_prefault_equilibrium"] is True


def test_dip_depth_monotonic_in_imax():
    imax_values = [0.3, 0.5, 0.8, 1.2, 1.8, 2.5, 4.0]
    dips = []
    for imax in imax_values:
        r = gcc.run_stress_test({"Imax": imax})
        assert r["baseline_ok"]
        dips.append(r["summary"]["min_v_poc"])
    diffs = np.diff(dips)
    assert np.all(diffs >= -1e-9), f"dip depth should be non-decreasing in Imax, got {dips}"


def test_imax_sweep_crosses_both_thresholds():
    sweep = gcc.run_imax_sweep()
    verdicts = [p["verdict"] for p in sweep["points"]]
    assert "NON-RECOVERABLE" in verdicts
    assert "RECOVERABLE" in verdicts


def test_baseline_pre_fault_equilibrium_is_small_signal_stable():
    r = gcc.run_stress_test({"Imax": 1.0})
    assert r["baseline"]["small_signal_stable"] is True
    assert r["baseline"]["max_eigenvalue_real_part"] < 0


def test_disclaimer_present():
    r = gcc.run_stress_test({})
    assert r.get("disclaimer")


def test_current_utilization_saturates_at_nonrecoverable_reference():
    """Real, verified physical story: current headroom insufficient -> the converter's own current
    utilization saturates to ~1.0 during the fault, directly explaining why the voltage dip is deep."""
    r = gcc.run_stress_test({"Imax": gcc.IMAX_NONRECOVERABLE_REFERENCE})
    assert r["summary"]["peak_current_utilization"] > 0.99
    assert r["summary"]["current_limit_active"] is True


def test_current_utilization_stays_below_limit_at_recoverable_reference():
    r = gcc.run_stress_test({"Imax": gcc.IMAX_RECOVERABLE_REFERENCE})
    assert r["summary"]["peak_current_utilization"] < 0.9
    assert r["summary"]["current_limit_active"] is False


def test_current_trace_never_exceeds_one_by_construction():
    """The clamp guarantees |i|/Imax <= 1 always -- verify this holds across the whole trace, not just the peak."""
    r = gcc.run_stress_test({"Imax": 0.4})
    assert max(r["current_trace"]) <= 1.0 + 1e-9
    """Importing/using this module must not mutate the Iberian scenario's frozen parameters."""
    from ims_platform.server import iberian_scenario as isc
    before = dict(isc.DEFAULT_NETWORK_PARAMS)
    gcc.run_stress_test({"Imax": 1.0})
    after = dict(isc.DEFAULT_NETWORK_PARAMS)
    assert before == after


def test_recommendation_engine_finds_critical_imax():
    r = gcc.find_recommended_interventions({"Imax": 0.4})
    assert r["baseline_verdict"] == "NON-RECOVERABLE"
    s = r["strategies"][0]
    assert s["achievable"] is True
    assert 1.0 < s["recommended_value"] < 1.6  # consistent with the verified boundary near Imax~1.38
    assert r["best_recommendation"] is not None


def test_recommendation_engine_honest_about_missing_levers():
    r = gcc.find_recommended_interventions({"Imax": 0.4})
    assert "not_applicable_levers" in r
    assert any("penetration" in lever for lever in r["not_applicable_levers"])


def test_recommendation_engine_already_recoverable_baseline():
    r = gcc.find_recommended_interventions({"Imax": 2.0})
    assert r["baseline_verdict"] == "RECOVERABLE"
