"""
Tests for the user-facing case interface (case.schema / case.loader / case.runner).
Run with: pytest tests/  (or python3 tests/test_case_interface.py)
"""
import os
import shutil
import numpy as np

try:
    import pytest
except ImportError:
    pytest = None

from ims_platform.case import Case, load_case, save_case_template, CaseRunner
from ims_platform.case.registry import MODEL_REGISTRY
from ims_platform.case.schema import CaseValidationError

TMP_DIR = "/tmp/ims_case_tests"


def setup_module(module=None):
    os.makedirs(TMP_DIR, exist_ok=True)


def teardown_module(module=None):
    shutil.rmtree(TMP_DIR, ignore_errors=True)


def test_case_from_minimal_dict_uses_defaults():
    case = Case.from_dict({"model": "grid_forming_inverter"})
    assert case.model == "grid_forming_inverter"
    assert case.recoverability.enabled is True
    assert case.control.enabled is False


def test_case_rejects_unknown_model():
    with pytest.raises(CaseValidationError) if pytest else _raises(CaseValidationError):
        Case.from_dict({"model": "not_a_real_model"})


def test_case_rejects_unknown_param():
    with pytest.raises(CaseValidationError) if pytest else _raises(CaseValidationError):
        Case.from_dict({"model": "grid_forming_inverter", "params": {"totally_bogus_param": 1.0}})


def test_save_and_load_yaml_template_roundtrip():
    path = os.path.join(TMP_DIR, "t1.yaml")
    save_case_template("grid_forming_inverter", path)
    assert os.path.exists(path)
    case = load_case(path)
    assert case.model == "grid_forming_inverter"


def test_save_and_load_json_template_roundtrip():
    path = os.path.join(TMP_DIR, "t2.json")
    save_case_template("dc_microgrid_cpl", path)
    assert os.path.exists(path)
    case = load_case(path)
    assert case.model == "dc_microgrid_cpl"


def test_load_case_missing_file_raises():
    with pytest.raises(FileNotFoundError) if pytest else _raises(FileNotFoundError):
        load_case(os.path.join(TMP_DIR, "does_not_exist.yaml"))


def test_runner_produces_report_and_plots_end_to_end():
    out_dir = os.path.join(TMP_DIR, "run1")
    case = Case.from_dict({
        "model": "grid_forming_inverter",
        "name": "unit-test-run",
        "nominal_input": [0.5],
        "state_guess": [0.3, 0.0],
        "manifold": {"sweep_param_range": [0.0, 0.9], "sweep_points": 20},
        "disturbance": {"type": "offset", "value": [0.3, 0.5], "t_horizon": 5.0},
        "recoverability": {"enabled": True, "radius": 1.0, "n_samples": 20, "t_horizon": 5.0},
        "control": {"enabled": False},
        "output": {"directory": out_dir, "formats": ["png", "md", "html"]},
    })
    results = CaseRunner(case, verbose=False).run()
    assert os.path.exists(os.path.join(out_dir, "report.md"))
    assert os.path.exists(os.path.join(out_dir, "report.html"))
    assert os.path.exists(os.path.join(out_dir, "01_manifold_trajectory.png"))
    assert 0.0 <= results["recoverability_open_loop"].recoverability_index <= 1.0


def test_runner_with_mrc_enabled_produces_closed_loop_report():
    out_dir = os.path.join(TMP_DIR, "run2")
    case = Case.from_dict({
        "model": "grid_forming_inverter",
        "name": "unit-test-mrc",
        "manifold": {"sweep_param_range": [0.0, 0.9], "sweep_points": 20},
        "disturbance": {"type": "offset", "value": [1.0, 2.0], "t_horizon": 5.0},
        "recoverability": {"enabled": True, "radius": 1.0, "n_samples": 15, "t_horizon": 5.0},
        "control": {"enabled": True, "Q_diag": [8.0, 2.0], "R_diag": [0.05]},
        "output": {"directory": out_dir, "formats": ["png", "md"]},
    })
    results = CaseRunner(case, verbose=False).run()
    assert "recoverability_closed_loop" in results
    assert os.path.exists(os.path.join(out_dir, "04_manifold_trajectory_MRC.png"))


def test_all_registered_models_have_metadata():
    from ims_platform.case.registry import MODEL_METADATA
    for key in MODEL_REGISTRY:
        assert key in MODEL_METADATA
        meta = MODEL_METADATA[key]
        assert "param_help" in meta and "default_operating_input" in meta and "default_state_guess" in meta


class _raises:
    """Minimal context-manager fallback for pytest.raises when pytest isn't installed."""
    def __init__(self, exc_type):
        self.exc_type = exc_type

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise AssertionError(f"expected {self.exc_type.__name__} to be raised")
        return issubclass(exc_type, self.exc_type)


if __name__ == "__main__":
    import sys
    import inspect
    setup_module()
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
    teardown_module()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
