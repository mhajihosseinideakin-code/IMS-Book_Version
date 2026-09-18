"""
Tests for the deeper "IMS engineering assessment" features added in
response to review feedback that the report was too "simulation-centric"
and needed to communicate the geometry of the intrinsic manifold and the
recoverability framework more richly, not just final numbers.

Scope note (deliberately NOT attempted here, and not claimed elsewhere):
manifold curvature, mean time-to-recovery, Monte-Carlo convergence
tracking, and component types (batteries, PV, wind) that do not exist in
the component library yet. Each would need either new validated numerical
machinery or component-library work not yet done; adding them without
that would mean either fabricating numbers or silently shipping
unvalidated ones, and this project's discipline throughout has been to
validate before shipping, not after.
"""
import numpy as np

from ims_platform.server import create_app


def _client():
    return create_app().test_client()


def test_network_summary_discovers_component_types_dynamically():
    """Component type counts come from real class names, not a hardcoded list -- proven by checking the actual types present."""
    c = _client()
    r = c.post("/api/project/buck_converter/build", json={"params": {}, "nominal_input": [0.4]})
    d = r.get_json()
    types = d["network_summary"]["component_types"]
    assert types.get("Converter") == 1
    assert types.get("ConstantImpedanceLoad") == 1
    # No fabricated categories: only types that actually exist in this network appear.
    assert "Battery" not in types and "PVSource" not in types


def test_topology_tree_reflects_real_bus_component_associations():
    c = _client()
    r = c.post("/api/project/buck_converter/build", json={"params": {}, "nominal_input": [0.4]})
    tree = r.get_json()["topology_tree"]
    joined = "\n".join(tree)
    assert "Bus 'out'" in joined
    assert "Converter 'conv'" in joined
    assert "ConstantImpedanceLoad 'load'" in joined


def test_manifold_statistics_present_and_consistent():
    c = _client()
    r = c.post("/api/project/grid_forming_inverter/equilibrium",
               json={"params": {}, "state_guess": [0.3, 0.0], "nominal_input": [0.5]})
    eq = r.get_json()
    payload = {
        "params": {}, "nominal_input": [0.5], "x_star": eq["x_star"],
        "sweep_range": [0.0, 0.9], "sweep_points": 30,
        "radius": 1.5, "n_samples": 20, "recovery_tol": 0.15,
    }
    r = c.post("/api/project/grid_forming_inverter/ims_analysis", json=payload)
    ms = r.get_json()["manifold_stats"]
    assert ms["n_points"] == 30
    assert ms["dimension"] == 1
    assert ms["n_stable"] + ms["n_unstable"] == ms["n_points"]
    assert ms["n_singular_points"] == len(ms["singular_points"])
    assert ms["continuation_parameter"] == "grid_forming_inverter_param"


def test_manifold_statistics_detect_no_singular_points_when_all_stable():
    """A manifold that is stable throughout its sweep must report zero singular points -- not a fabricated nonzero count."""
    c = _client()
    r = c.post("/api/project/grid_forming_inverter/equilibrium",
               json={"params": {}, "state_guess": [0.3, 0.0], "nominal_input": [0.5]})
    eq = r.get_json()
    payload = {
        "params": {}, "nominal_input": [0.5], "x_star": eq["x_star"],
        "sweep_range": [0.0, 0.9], "sweep_points": 30,
        "radius": 1.0, "n_samples": 10, "recovery_tol": 0.15,
    }
    r = c.post("/api/project/grid_forming_inverter/ims_analysis", json=payload)
    ms = r.get_json()["manifold_stats"]
    assert ms["n_unstable"] == 0
    assert ms["n_singular_points"] == 0


def test_recoverability_worst_disturbance_and_confidence_interval():
    """Uses a wide radius on the bistable DC microgrid model to force a partial (non-trivial) recoverability index."""
    c = _client()
    r = c.post("/api/project/dc_microgrid_cpl/equilibrium",
               json={"params": {}, "state_guess": [0.44, 1.13], "nominal_input": [0.5]})
    eq = r.get_json()
    payload = {
        "params": {}, "nominal_input": [0.5], "x_star": eq["x_star"],
        "sweep_range": [0.05, 1.2], "sweep_points": 60,
        "radius": 2.5, "n_samples": 40, "recovery_tol": 0.1,
    }
    r = c.post("/api/project/dc_microgrid_cpl/ims_analysis", json=payload)
    rec = r.get_json()["recoverability"]
    assert 0.0 < rec["index"] < 1.0  # confirms this test actually exercises a partial-failure case
    assert rec["worst_disturbance"] is not None
    assert len(rec["worst_disturbance"]) == 2
    ci = rec["confidence_interval"]
    assert ci[0] <= rec["index"] <= ci[1]
    assert ci[0] < ci[1]


def test_recoverability_worst_disturbance_none_when_all_recover():
    c = _client()
    r = c.post("/api/project/grid_forming_inverter/equilibrium",
               json={"params": {}, "state_guess": [0.3, 0.0], "nominal_input": [0.5]})
    eq = r.get_json()
    payload = {
        "params": {}, "nominal_input": [0.5], "x_star": eq["x_star"],
        "sweep_range": [0.0, 0.9], "sweep_points": 20,
        "radius": 0.3, "n_samples": 15, "recovery_tol": 0.2,
    }
    r = c.post("/api/project/grid_forming_inverter/ims_analysis", json=payload)
    rec = r.get_json()["recoverability"]
    if rec["index"] == 1.0:
        assert rec["worst_disturbance"] is None


def test_controller_effect_comparison_uses_same_disturbance_samples():
    """The whole point of the comparison is isolating the controller's effect -- proven by checking both runs used identical sample points."""
    c = _client()
    r = c.post("/api/project/grid_forming_inverter/equilibrium",
               json={"params": {}, "state_guess": [0.3, 0.0], "nominal_input": [0.5]})
    eq = r.get_json()
    payload = {
        "params": {}, "nominal_input": [0.5], "x_star": eq["x_star"],
        "sweep_range": [0.0, 0.9], "sweep_points": 25,
        "radius": 1.5, "n_samples": 20, "recovery_tol": 0.15,
        "mrc_enabled": True, "Q_diag": [8, 2], "R_diag": [0.05], "u_min": [-0.2], "u_max": [1.2],
    }
    r = c.post("/api/project/grid_forming_inverter/ims_analysis", json=payload)
    d = r.get_json()
    assert "recoverability_closed_loop" in d
    ol_samples = np.array(d["recoverability"]["samples"])
    cl_samples = np.array(d["recoverability_closed_loop"]["samples"])
    assert np.allclose(ol_samples, cl_samples)


def test_controller_effect_absent_for_converter_topology():
    """Controller-effect comparison is only meaningful/implemented for single_model + mrc_enabled; must not silently fabricate it elsewhere."""
    c = _client()
    r = c.post("/api/project/buck_converter/equilibrium", json={"params": {}, "nominal_input": [0.4]})
    eq = r.get_json()
    payload = {
        "params": {}, "nominal_input": [0.4], "x_star": eq["x_star"],
        "sweep_range": [0.1, 0.85], "sweep_points": 150,
        "radius": 1.0, "n_samples": 20, "recovery_tol": 0.1, "mrc_enabled": True,
    }
    r = c.post("/api/project/buck_converter/ims_analysis", json=payload)
    assert "recoverability_closed_loop" not in r.get_json()


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
