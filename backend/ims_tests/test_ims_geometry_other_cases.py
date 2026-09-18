"""
Tests for server.ims_geometry.check_manifold_applicability and its use
in server.gfm_current_limit_case / server.multi_converter_fault_case's
own run_ims_geometry_analysis functions.

These specifically guard against assuming the Iberian scenario's
manifold construction "just works" for other cases merely because they
share GFLRenewableSource -- the explicit requirement this file exists
to enforce.
"""
import numpy as np

from ims_platform.server import iberian_scenario as isc
from ims_platform.server import gfm_current_limit_case as gcc
from ims_platform.server import multi_converter_fault_case as mcc
from ims_platform.server import ims_geometry as img


def test_iberian_manifold_construction_is_applicable():
    system, hist, used = isc._build_system({"KQ": 0.0, "Q_avail": 0.05, "Imax": 1.3})
    gfl = img.find_gfl_component(system, "gfl_sw")
    check = img.check_manifold_applicability(gfl, v_probe_range=(0.8, 1.4))
    assert check["applicable"] is True
    assert check["timer_star_range_over_probe"] > 0.1


def test_gfm_case_manifold_construction_is_genuinely_degenerate():
    """
    Verified finding, not assumed: the GFM Current-Limit Recovery case's
    converter has v_trip=999 (overvoltage protection intentionally
    disabled), so timer*(v) is constant at 0 across the entire realistic
    operating range.
    """
    system, comps = gcc.build_network(Imax=0.4, P_nom=0.3)
    gfm = img.find_gfl_component(system, "gfm_conv")
    check = img.check_manifold_applicability(gfm, v_probe_range=(0.3, 1.5))
    assert check["applicable"] is False
    assert check["timer_star_range_over_probe"] < 1e-6
    assert all(abs(v) < 1e-9 for v in check["timer_star_values_probed"])


def test_multiconverter_case_both_converters_are_degenerate():
    """Checked independently for BOTH converters -- not assumed from the GFM case's result."""
    system, comps = mcc.build_network(1.0, 1.0, 0.0)
    gfm = img.find_gfl_component(system, "conv_gfm")
    gfl = img.find_gfl_component(system, "conv_gfl")
    check_gfm = img.check_manifold_applicability(gfm, v_probe_range=(0.3, 1.5))
    check_gfl = img.check_manifold_applicability(gfl, v_probe_range=(0.3, 1.5))
    assert check_gfm["applicable"] is False
    assert check_gfl["applicable"] is False


def test_gfm_run_ims_geometry_analysis_reports_not_applicable_honestly():
    r = gcc.run_ims_geometry_analysis({"Imax": 0.4})
    assert r["available"] is False
    assert "NOT APPLICABLE" in r["message"]
    assert "v_trip=999" in r["message"]
    assert "applicability_check" in r
    # Evidence must be present, not just an assertion.
    assert len(r["applicability_check"]["timer_star_values_probed"]) > 0


def test_multiconverter_run_ims_geometry_analysis_reports_not_applicable_for_both():
    r = mcc.run_ims_geometry_analysis(mcc.BASELINE_REFERENCE)
    assert r["available"] is False
    assert "applicability_check_gfm" in r and "applicability_check_gfl" in r
    assert r["applicability_check_gfm"]["applicable"] is False
    assert r["applicability_check_gfl"]["applicable"] is False


def test_applicability_check_would_detect_a_hypothetical_applicable_gfm_case():
    """
    Sanity check on the check itself: if a GFM-style component DID have
    a realistic v_trip (not the disabled sentinel), the check should
    correctly report it as applicable -- confirms this isn't just
    hard-coded to always say "not applicable" for any component named
    similarly to gfm_conv/conv_gfm.
    """
    from ims_platform.network import GFLRenewableSource
    comp = GFLRenewableSource(id="hypothetical", bus="b", P_set=0.3, KQ=0.0, Imax=1.0,
                               v_trip=1.10, t_delay=0.15, gfm_fraction=1.0)
    check = img.check_manifold_applicability(comp, v_probe_range=(0.8, 1.4))
    assert check["applicable"] is True
