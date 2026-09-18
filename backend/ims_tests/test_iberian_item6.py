"""
Tests for the Item 6 backend functions in server.iberian_scenario:
2-D recoverability boundary map, GFM-penetration comparison (reported
honestly across severities, not just the full historical one),
pre-incident warning lead-time diagnostic, the intervention
recommendation engine, and the Case Library registry.

As with tests/test_iberian_scenario.py and tests/test_gfm_mechanism.py,
these check REAL simulation outputs against physically-motivated
expectations, not fixed "golden" numbers that would make a future
honest re-tuning look like a regression.
"""
import numpy as np

try:
    import pytest
except ImportError:
    pytest = None

from ims_platform.server import iberian_scenario as isc


def test_2d_boundary_map_shape_and_monotonicity():
    bm = isc.run_boundary_map_2d({"Q_avail": 0.05, "Imax": 1.3}, x_param="KQ",
                                  x_values=[0.0, 8.0, 16.0], severities=[0.2, 0.6, 1.0])
    assert bm["x_param"] == "KQ"
    assert len(bm["grid"]) == 3
    assert all(len(row) == 3 for row in bm["grid"])

    verdict_rank = {"RECOVERABLE": 2, "AT RISK": 1, "NON-RECOVERABLE": 0, None: -1}
    # More KQ, at fixed severity, should never make things WORSE (non-decreasing recoverability).
    for j in range(3):
        col = [verdict_rank[bm["grid"][i][j]["verdict"]] for i in range(3)]
        assert all(col[i] <= col[i + 1] for i in range(len(col) - 1)), f"KQ column {j} not monotonic: {col}"


def test_2d_boundary_map_rejects_unknown_x_param():
    if pytest is not None:
        with pytest.raises(ValueError):
            isc.run_boundary_map_2d({}, x_param="not_a_real_param")


def test_gfm_comparison_matches_frozen_finding_at_full_severity():
    """At full severity, 0/20/40% GFM should show no meaningful difference (the honest, verified finding)."""
    cmp = isc.run_gfm_comparison({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3},
                                  gfm_fractions=[0.0, 0.2, 0.4], severities=[1.0])
    verdicts_at_full = [row["points"][0]["verdict"] for row in cmp["rows"]]
    assert all(v == "NON-RECOVERABLE" for v in verdicts_at_full), (
        f"expected no recoverability change at full severity, got {verdicts_at_full} "
        "-- if this is a deliberate, reviewed physics change, update this test consciously"
    )


def test_gfm_comparison_shows_real_effect_at_moderate_severity():
    cmp = isc.run_gfm_comparison({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3},
                                  gfm_fractions=[0.0, 0.4], severities=[0.8])
    v0 = cmp["rows"][0]["points"][0]["verdict"]
    v40 = cmp["rows"][1]["points"][0]["verdict"]
    assert v0 == "NON-RECOVERABLE"
    assert v40 == "RECOVERABLE"


def test_pre_incident_warning_genuinely_leads_on_nonrecoverable_case():
    r = isc.run_stress_test({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3}, severity=1.0, tail_horizon=8.0)
    w = isc.analyze_pre_incident_warning(r)
    assert w["available"]
    assert w["genuinely_leads"] is True
    assert w["lead_time_seconds"] > 0


def test_pre_incident_warning_absent_on_recoverable_case():
    """A case that never exceeds v_trip should report no crossing at all, not a fabricated lead."""
    r = isc.run_stress_test({"KQ": 15.0, "Q_avail": 0.05, "Imax": 1.3}, severity=1.0, tail_horizon=8.0)
    w = isc.analyze_pre_incident_warning(r)
    assert w["available"]
    assert w["t_overvoltage_trip"] is None
    assert w["genuinely_leads"] is False


def test_pre_incident_warning_lead_grows_with_slower_disturbance():
    """Physically sensible: a slower/milder disturbance should give more warning lead time, not less."""
    r_full = isc.run_stress_test({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3}, severity=1.0, tail_horizon=8.0)
    r_moderate = isc.run_stress_test({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3}, severity=0.8, tail_horizon=8.0)
    w_full = isc.analyze_pre_incident_warning(r_full)
    w_moderate = isc.analyze_pre_incident_warning(r_moderate)
    assert w_full["genuinely_leads"] and w_moderate["genuinely_leads"]
    assert w_moderate["lead_time_seconds"] > w_full["lead_time_seconds"]


def test_recommendation_engine_finds_known_achievable_and_unachievable_strategies():
    rec = isc.find_recommended_interventions({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3}, severity=1.0)
    assert rec["baseline_verdict"] == "NON-RECOVERABLE"
    by_label = {s["label"]: s for s in rec["strategies"]}

    kq_alone = by_label["Dynamic reactive-power support (KQ) alone"]
    assert kq_alone["achievable"]
    assert 8.0 < kq_alone["recommended_value"] < 13.0  # consistent with the verified ~KQ=10-11 bifurcation

    qavail_alone = by_label["Shunt-reactor availability alone"]
    assert qavail_alone["achievable"] is False  # verified: best case alone is AT RISK, not RECOVERABLE

    combo = by_label["Full shunt-reactor availability + grid-forming penetration"]
    assert combo["achievable"]
    assert combo["recommended_value"] < gfm_alone_upper_bound_sanity_check()

    assert rec["best_recommendation"] is not None


def gfm_alone_upper_bound_sanity_check():
    return 1.0  # combined-strategy gfm_fraction must be a valid fraction


def test_recommendation_engine_reports_already_recoverable_baseline_honestly():
    rec = isc.find_recommended_interventions({"KQ": 15.0, "Q_avail": 0.05, "Imax": 1.3}, severity=1.0)
    assert rec["baseline_verdict"] == "RECOVERABLE"


def test_counterfactual_uses_only_kq_as_the_differing_variable():
    r = isc.run_baseline_vs_dynamic_q_counterfactual()
    assert r["baseline"]["params_used"]["Q_avail"] == r["dynamic_q"]["params_used"]["Q_avail"]
    assert r["baseline"]["params_used"]["Imax"] == r["dynamic_q"]["params_used"]["Imax"]
    assert r["baseline"]["params_used"]["gfm_fraction"] == r["dynamic_q"]["params_used"]["gfm_fraction"]
    assert r["baseline"]["params_used"]["KQ"] != r["dynamic_q"]["params_used"]["KQ"]
    assert r["baseline"]["summary"]["verdict"] == "NON-RECOVERABLE"
    assert r["dynamic_q"]["summary"]["verdict"] == "RECOVERABLE"
    assert len(r["comparison_table"]) == 5


def test_case_library_structure():
    ids = [c["id"] for c in isc.CASE_LIBRARY]
    assert len(ids) == len(set(ids)), "case ids must be unique"
    assert "iberian_2025_overvoltage_cascade" in ids
    for case in isc.CASE_LIBRARY:
        assert {"id", "name", "description", "available", "kind"} <= set(case.keys())
        # Every "available" case must be something the API can actually run --
        # either the Iberian scenario itself, or an existing built-in project id.
        if case["available"]:
            assert case["kind"] in ("iberian_scenario", "single_model", "converter_topology",
                                     "gfm_current_limit", "multi_converter_fault")


def test_all_five_proposal_cases_are_present_and_active():
    """Full Case Library scope: all five proposal-defined cases present, none left as a roadmap placeholder."""
    expected_ids = {
        "iberian_2025_overvoltage_cascade",
        "dc_microgrid_cpl",
        "grid_forming_inverter",
        "gfm_current_limit_recovery",
        "multi_converter_fault_recovery",
    }
    ids = {c["id"] for c in isc.CASE_LIBRARY}
    assert expected_ids <= ids
    for case in isc.CASE_LIBRARY:
        if case["id"] in expected_ids:
            assert case["available"] is True, f"{case['id']} is not active"
