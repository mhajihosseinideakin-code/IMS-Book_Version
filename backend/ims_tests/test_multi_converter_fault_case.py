"""Tests for server.multi_converter_fault_case -- the "Multi-Converter Fault Recovery" Case Library entry."""
import numpy as np

from ims_platform.server import multi_converter_fault_case as mcc


def test_reference_bifurcation():
    r = mcc.verify_reference_bifurcation()
    assert r["baseline_case"]["summary"]["verdict"] == "NON-RECOVERABLE"
    assert r["coordinated_case"]["summary"]["verdict"] == "RECOVERABLE"


def test_local_vs_remote_support_asymmetry_is_genuine():
    """
    The corrected, verified finding: local (grid-following) support is
    necessary and sufficient; remote (grid-forming anchor) support
    alone is not. This test locks in the ACTUAL result, not the
    "coordination is required" assumption an earlier draft of this
    case made before checking.
    """
    r = mcc.verify_local_vs_remote_support_asymmetry()
    assert r["gfm_only"]["summary"]["verdict"] == "NON-RECOVERABLE"
    assert r["gfl_only"]["summary"]["verdict"] == "RECOVERABLE"
    assert r["both"]["summary"]["verdict"] == "RECOVERABLE"


def test_two_converters_have_independent_states():
    """Confirms this is genuinely a two-converter model (distinct states/buses), not a duplicated single model."""
    system, comps = mcc.build_network(Imax_gfm=1.0, Imax_gfl=1.0, KQ_gfl=0.0)
    assert comps["gfm"] is not comps["gfl"]
    assert comps["gfm"].bus != comps["gfl"].bus
    assert "v_busA" in system.state_names
    assert "v_busB" in system.state_names
    assert "conv_gfm_trip_timer" in system.state_names
    assert "conv_gfl_trip_timer" in system.state_names


def test_every_fault_returns_to_same_prefault_equilibrium():
    for params in [mcc.BASELINE_REFERENCE, mcc.COORDINATED_REFERENCE]:
        r = mcc.run_stress_test(params)
        assert r["baseline_ok"]
        assert r["summary"]["settled_to_prefault_equilibrium"] is True


def test_baseline_prefault_equilibrium_small_signal_stable():
    r = mcc.run_stress_test(mcc.BASELINE_REFERENCE)
    assert r["baseline"]["small_signal_stable"] is True


def test_limiting_bus_is_reported():
    r = mcc.run_stress_test(mcc.BASELINE_REFERENCE)
    assert r["summary"]["limiting_bus"] in ("busA", "busB")


def test_disclaimer_present():
    r = mcc.run_stress_test({})
    assert r.get("disclaimer")


def test_gfm_anchor_current_saturates_at_baseline_reference():
    """Real finding: the GFM anchor's OWN current genuinely saturates at the weak baseline settings,
    even though its remote effect on busB is weak -- these are two different, both-true facts."""
    r = mcc.run_stress_test(mcc.BASELINE_REFERENCE)
    assert r["summary"]["peak_current_utilization_gfm"] > 0.99


def test_current_traces_never_exceed_one_by_construction():
    r = mcc.run_stress_test(mcc.BASELINE_REFERENCE)
    assert max(r["current_traces"]["gfm"]) <= 1.0 + 1e-9
    assert max(r["current_traces"]["gfl"]) <= 1.0 + 1e-9


def test_does_not_touch_iberian_or_gfm_case_frozen_params():
    from ims_platform.server import iberian_scenario as isc
    from ims_platform.server import gfm_current_limit_case as gcc
    before_iberian = dict(isc.DEFAULT_NETWORK_PARAMS)
    before_gfm = dict(gcc.DEFAULT_PARAMS)
    mcc.run_stress_test(mcc.BASELINE_REFERENCE)
    assert dict(isc.DEFAULT_NETWORK_PARAMS) == before_iberian
    assert dict(gcc.DEFAULT_PARAMS) == before_gfm


def test_recommendation_engine_baseline_and_limiting_converter():
    r = mcc.find_recommended_interventions(mcc.BASELINE_REFERENCE)
    assert r["baseline_verdict"] == "NON-RECOVERABLE"
    assert r["limiting_converter"] == "busB"


def test_recommendation_engine_single_levers_alone_insufficient_at_default_baseline():
    """Verified finding (not assumed): with all OTHER params at true defaults, no single lever
    alone recovers this disturbance -- only the coordinated combination does."""
    r = mcc.find_recommended_interventions(mcc.BASELINE_REFERENCE)
    by_label = {s["label"]: s for s in r["strategies"]}
    assert by_label["Increase GFM anchor current limit (Imax_gfm) alone"]["achievable"] is False
    assert by_label["Increase GFL follower current limit (Imax_gfl) alone"]["achievable"] is False
    assert by_label["Increase GFL follower reactive support (KQ_gfl) alone"]["achievable"] is False
    combo = by_label["Coordinated: full current headroom on both converters + GFL reactive support"]
    assert combo["achievable"] is True
    assert r["best_recommendation"] is not None


def test_recommendation_engine_already_recoverable_baseline():
    r = mcc.find_recommended_interventions(mcc.COORDINATED_REFERENCE)
    assert r["baseline_verdict"] == "RECOVERABLE"
