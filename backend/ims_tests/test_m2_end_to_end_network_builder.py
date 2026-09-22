"""
ims_tests.test_m2_end_to_end_network_builder
----------------------------------------------

"Final M2 Cleanup and End-to-End Validation" -- item 2 ("End-to-End
Network Builder Validation") and item 3/4 ("Report Transparency" /
"Baseline vs M1 vs M2") run through the ACTUAL Network Builder /
server.app endpoints (network_mrc_inspect, network_mrc_auto_apply,
network_mrc_auto_closed_loop, run_auto_mrc_pipeline_multi) -- not just
the direct mrc_designer.auto_manifold backend functions -- so that a
genuine backend/report/UI disagreement (like the ones found and fixed
here) would actually be caught.

Bugs found and fixed by this validation pass (all in code, not just
tests):

  1. run_auto_mrc_pipeline_multi's per-converter feasibility check
     (auto_manifold.py) only counted "MRC_SYNTHESIS_SUPPORTED" and
     "MRC_FEASIBILITY_DIAGNOSTIC_ONLY" as feasible -- a converter whose
     only viable path is the M2/reshaped-manifold outcome
     ("MRC_SYNTHESIS_SUPPORTED_VIA_RESHAPED_MANIFOLD") was silently
     treated as NOT feasible in a multi-converter scan, disagreeing with
     what run_auto_mrc_pipeline itself reports for that exact converter.

  2. server/app.py's network_mrc_inspect computed 'mrc_established' from
     only the phi1 SUPPORTED status, so a genuinely-established M2
     result was reported as NOT established to callers (and to the
     explorer.html UI, which reads this field directly).

  3. server/app.py's network_mrc_auto_apply and network_mrc_auto_closed_loop
     refused to operate at all on the M2 status -- there was previously
     NO way to actually attach the reshaped-manifold controller, or run
     a closed-loop comparison for it, through the Network Builder
     workflow, even though the pipeline had already determined synthesis
     was supported.

  4. explorer.html's customMrcStatusKey() had the identical status-string
     gap as (2)/(3): the UI rendered "MRC SYNTHESIS NOT ESTABLISHED" for
     a network the backend had actually accepted via M2 (see
     test_explorer_report_content.js for the JS-side regression check on
     the report generator; this file exercises the Python backend that
     JS consumes).
"""

import pytest

from ims_platform.server.app import (
    _build_network_from_spec,
    network_mrc_inspect,
    network_mrc_auto_apply,
    network_mrc_auto_closed_loop,
)
from ims_platform.mrc_designer.auto_manifold import run_auto_mrc_pipeline_multi


def _spec(topology, P, v_nom, v_in=48.0, L=1e-3, R_L=0.05, C=2e-3, extra_converters=None):
    conv = {"id": "conv", "bus": "bus", "topology": topology,
            "params": {"v_in": v_in, "L": L, "R_L": R_L},
            "controller": "constant_duty", "controller_params": {"d": 0.5}}
    converters = [conv] + (extra_converters or [])
    return {
        "name": "t", "buses": [{"id": "bus", "type": "dynamic", "C": C, "v_init": v_nom}],
        "converters": converters,
        "loads": [{"id": "load", "bus": "bus", "type": "cpl", "P": P, "v_floor": 0.5}],
        "input_component_id": "load",
    }


# ---------------------------------------------------------------------------
# End-to-end representative cases through the real workflow.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("topology,v_nom", [("buck", 24.0), ("boost", 72.0), ("buckboost", 24.0)])
def test_cpl_case_reaches_m2_through_the_real_endpoints_and_stays_consistent(topology, v_nom):
    """Network Builder -> Model Assembly -> IMS/MRC design (/inspect) ->
    M2 reshaping -> Apply (/auto_apply) -> Closed-loop verification
    (/auto_closed_loop), for a CPL case where phi1 (M1) is known
    UNSTABLE and only M2 is feasible. Verifies the UI-facing 'inspect'
    flag agrees with the pipeline's own status, that /auto_apply attaches
    the ACTUAL M2 controller (not phi1), and that /auto_closed_loop
    returns a genuine baseline/M1/M2 comparison consistent with the
    reshaping report."""
    spec = _spec(topology, P=50.0, v_nom=v_nom)
    payload = {"network_spec": spec, "converter_id": "conv", "v_nom": v_nom}

    inspect = network_mrc_inspect(payload)
    assert inspect["establishment_basis"] == "MRC_SYNTHESIS_SUPPORTED_VIA_RESHAPED_MANIFOLD"
    # Bug (2): this must be True -- an M2-established network is genuinely established.
    assert inspect["mrc_established"] is True
    assert inspect["classification"] == "MRC SYNTHESIS SUPPORTED — RESHAPED MANIFOLD"

    applied = network_mrc_auto_apply(payload)
    assert applied["status"] == "MRC_SYNTHESIS_SUPPORTED_VIA_RESHAPED_MANIFOLD"
    # Bug (3): must attach the SECOND candidate's controller, never phi1's.
    assert applied["converter"]["active_controller"]["type"] == "BusVoltageAnchoredMRCController"
    assert applied["converter"]["active_controller"]["candidate"] == "phi2 (bus-voltage-anchored, reshaped)"

    cl = network_mrc_auto_closed_loop(payload)
    assert cl["status"] == "MRC_SYNTHESIS_SUPPORTED_VIA_RESHAPED_MANIFOLD"
    cmp = cl["baseline_m1_m2_comparison"]
    assert cmp["baseline"]["equilibrium_converged"]
    assert cmp["m1"]["equilibrium_converged"]
    assert cmp["m2"]["equilibrium_converged"]
    # M1 (current-anchored, unstable reduced dynamics) must NOT recover;
    # M2 (reshaped, both transverse and reduced dynamics stable) must.
    assert cmp["m1"]["recovered"] is False
    assert cmp["m2"]["recovered"] is True
    # closed_loop_comparison stays baseline-vs-active-controller shaped
    # (backward compatible with the existing chart/report renderers).
    assert set(cl["closed_loop_comparison"].keys()) == {"baseline", "mrc"}
    assert cl["closed_loop_comparison"]["mrc"] is cmp["m2"]


def test_impedance_load_case_where_m1_already_suffices_is_unaffected():
    """An impedance load at the natural M1-stabilizing R (chosen so phi1's
    reduced dynamics are already stable) must go through the ORIGINAL
    (M1, current-anchored) path end-to-end, completely unaffected by the
    M2 additions -- the pre-existing behavior is a preservation
    requirement, not just a new feature."""
    spec = _spec("buck", P=0.0, v_nom=24.0)
    spec["loads"] = [{"id": "load", "bus": "bus", "type": "impedance", "R": 11.52}]
    spec.pop("input_component_id", None)  # only 'conv' has an exogenous input now -- unambiguous
    payload = {"network_spec": spec, "converter_id": "conv", "v_nom": 24.0}

    inspect = network_mrc_inspect(payload)
    assert inspect["establishment_basis"] == "MRC_SYNTHESIS_SUPPORTED"
    assert inspect["mrc_established"] is True

    applied = network_mrc_auto_apply(payload)
    assert applied["converter"]["active_controller"]["type"] == "AutoCurrentLoopMRCController"
    assert applied["converter"]["active_controller"]["candidate"] == "phi1 (current-anchored)"

    cl = network_mrc_auto_closed_loop(payload)
    assert cl["status"] == "MRC_SYNTHESIS_SUPPORTED"
    assert "baseline_m1_m2_comparison" not in cl
    assert set(cl["closed_loop_comparison"].keys()) == {"baseline", "mrc", "v_nom"}


def test_multi_converter_case_counts_m2_only_feasible_converter_as_feasible():
    """Bug (1): a network with two duty-modulated converters, one of
    which is ONLY feasible via M2 reshaping, must have that converter
    correctly marked feasible=True in the per-converter scan -- not
    silently dropped because the scan only recognized the two phi1-only
    status strings."""
    spec = _spec("buck", P=50.0, v_nom=24.0,
                 extra_converters=[{"id": "conv2", "bus": "bus2", "topology": "buck",
                                     "params": {"v_in": 48.0, "L": 1e-3, "R_L": 0.05},
                                     "controller": "constant_duty", "controller_params": {"d": 0.5}}])
    spec["buses"].append({"id": "bus2", "type": "dynamic", "C": 2e-3, "v_init": 24.0})
    spec["loads"].append({"id": "load2", "bus": "bus2", "type": "impedance", "R": 11.52})

    net, _input_id = _build_network_from_spec(spec)
    report = run_auto_mrc_pipeline_multi(net, converter_id=None, v_nom=24.0)

    assert report["status"] == "MRC_MULTI_CANDIDATE_SELECTION_REQUIRED"
    scan_by_id = {pc["converter_id"]: pc for pc in report["multi_converter_scan"]}
    assert scan_by_id["conv"]["status"] == "MRC_SYNTHESIS_SUPPORTED_VIA_RESHAPED_MANIFOLD"
    # This is exactly the case bug (1) got wrong: feasible must be True here.
    assert scan_by_id["conv"]["feasible"] is True
    assert scan_by_id["conv"]["reshaped_manifold"] is True
    assert scan_by_id["conv2"]["status"] == "MRC_SYNTHESIS_SUPPORTED"
    assert scan_by_id["conv2"]["feasible"] is True
    assert scan_by_id["conv2"]["reshaped_manifold"] is False
    assert set(report["feasible_converter_ids"]) == {"conv", "conv2"}
