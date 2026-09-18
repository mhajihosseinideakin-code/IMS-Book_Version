"""
Automated tests for the shared IMS analysis framework
(ims_platform.framework) and the standardized end-to-end run of the
Four-State Stabilising-MRC through the framework interface.

Run: cd /app/backend && PYTHONPATH=/app/backend \
     python -m pytest ims_tests/test_analysis_framework.py -o addopts="" -q
"""
import numpy as np
import pytest

from ims_platform.framework import list_descriptors, get_analysis, STANDARD_WORKFLOW_STAGES
from ims_platform.framework.registry import get_descriptor
from ims_platform.framework.interfaces import Analysis, SupportedOutputs

EXPECTED_STAGES = ["Model/Network", "Analysis", "Disturbance", "Simulation", "IMS Results", "Report"]


def test_standard_workflow_stages():
    assert STANDARD_WORKFLOW_STAGES == EXPECTED_STAGES


def test_registry_lists_mrc_and_all_case_library_studies():
    ids = [d["id"] for d in list_descriptors()]
    assert "stabilizing_mrc" in ids
    for case_id in ["iberian_2025_overvoltage_cascade", "dc_microgrid_cpl",
                    "grid_forming_inverter", "gfm_current_limit_recovery",
                    "multi_converter_fault_recovery"]:
        assert case_id in ids, f"missing case {case_id}"


def test_descriptors_declare_supported_outputs_and_entry():
    for d in list_descriptors():
        assert "supported_outputs" in d and isinstance(d["supported_outputs"], dict)
        assert set(SupportedOutputs().to_dict()).issubset(d["supported_outputs"])
        assert d["workflow_stages"] == EXPECTED_STAGES
        # standardized analyses are runnable; cases carry a legacy entry route
        if not d["standardized"]:
            assert d["entry"].get("view_fn"), f"case {d['id']} has no entry.view_fn"


def test_mrc_conforms_to_analysis_protocol():
    mrc = get_analysis("stabilizing_mrc")
    assert isinstance(mrc, Analysis)  # runtime_checkable protocol
    assert mrc.descriptor().standardized is True
    assert mrc.descriptor().supported_outputs.equilibrium is True


def test_cases_are_not_runnable_but_are_discoverable():
    # cases are registered as descriptors, not runnable analyses (yet)
    with pytest.raises(KeyError):
        get_analysis("iberian_2025_overvoltage_cascade")
    d = get_descriptor("iberian_2025_overvoltage_cascade")
    assert d["standardized"] is False
    assert d["title"].startswith("Iberian")


def test_mrc_validate_rejects_invalid_config():
    mrc = get_analysis("stabilizing_mrc")
    cfg = mrc.default_config()
    cfg["params"]["L"] = -1.0
    errs = mrc.validate(cfg)
    assert errs and any("L must be > 0" in e for e in errs)


def test_mrc_run_produces_standardized_envelope():
    mrc = get_analysis("stabilizing_mrc")
    env = mrc.run(mrc.default_config())
    # envelope shape
    for key in ["analysis_id", "supported_outputs", "equilibrium", "manifold",
                "residual_trajectory", "contraction_evidence", "local_stability",
                "recovery_status", "solver_status", "trajectories", "claim_levels", "run_id"]:
        assert key in env, f"missing envelope key {key}"
    # numerical benchmark preserved through the framework path
    assert env["equilibrium"]["residual_norm"] < 1e-9
    assert env["local_stability"]["transverse_check_ok"] is True
    assert env["contraction_evidence"]["max_rel_error"] < 1e-4
    # claim levels kept distinct and correctly attached
    assert env["contraction_evidence"]["claim_level"] == "ideal_residual_contraction"
    assert env["local_stability"]["claim_level"] == "local_equilibrium_stability"
    assert env["recovery_status"]["claim_level"] == "observed_finite_horizon_recovery"
    assert "certified_regional_ims" in env["claim_levels"]


def test_mrc_run_matches_direct_workflow_numbers():
    # framework path must equal the verified direct workflow (no divergence)
    from ims_platform.stabilizing import compute_stability
    mrc = get_analysis("stabilizing_mrc")
    cfg = mrc.default_config()
    env = mrc.run(cfg)
    st = compute_stability(cfg["params"])
    fw = sorted(e["re"] for e in env["local_stability"]["full_eigenvalues"])
    direct = sorted(e["re"] for e in st["full_eigenvalues"])
    assert np.allclose(fw, direct, atol=0, rtol=0)
