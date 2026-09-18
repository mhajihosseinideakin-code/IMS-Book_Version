"""
Tests for the IMS Platform Explorer's project registry and staged
endpoints (server.app: build / equilibrium / simulate / ims_analysis),
covering all three architectural categories the six built-in projects
span: single_model, converter_topology, and network_mrc.

Two real bugs were found and fixed while building this: a default
simulation horizon (20ms) shorter than the buck/boost/buck-boost
converters' own ~28ms LC oscillation period, and a manifold sweep
resolution (30 points) coarse enough that the manifold residual AT THE
TRUE EQUILIBRIUM ITSELF exceeded the recoverability tolerance -- neither
caught by "does it return 200", both caught by checking the actual
numbers against what they should physically be.
"""
import numpy as np

from ims_platform.server import create_app
from ims_platform.server.app import PROJECTS


def _client():
    return create_app().test_client()


def test_projects_list_has_six_entries_across_three_categories():
    r = _client().get("/api/projects")
    assert r.status_code == 200
    data = r.get_json()
    assert len(data) == 6
    categories = {p["category"] for p in data}
    assert categories == {"single_model", "converter_topology", "network_mrc"}


def test_project_detail_unknown_id_404():
    r = _client().get("/api/project/does_not_exist")
    assert r.status_code == 404


def test_single_model_full_staged_pipeline():
    c = _client()
    pid = "grid_forming_inverter"
    payload = {"params": {}, "state_guess": [0.3, 0.0], "nominal_input": [0.5]}

    r = c.post(f"/api/project/{pid}/build", json=payload)
    assert r.status_code == 200
    assert r.get_json()["state_names"] == ["delta", "omega"]

    r = c.post(f"/api/project/{pid}/equilibrium", json=payload)
    assert r.status_code == 200
    eq = r.get_json()
    assert abs(eq["x_star"][0] - 0.1506) < 1e-3
    assert eq["stable"] is True

    payload2 = dict(payload, x_star=eq["x_star"], disturbance={"type": "offset", "value": [1.0, 2.0]}, horizon=10.0, mrc_enabled=True)
    r = c.post(f"/api/project/{pid}/simulate", json=payload2)
    assert r.status_code == 200
    sim = r.get_json()
    assert sim["controller_info"]["type"] == "Scheduled LQR (Conventional Control Library)"
    assert sim["trajectory_closed_loop"]["success"]

    payload3 = dict(payload2, sweep_range=[0.0, 0.9], sweep_points=30, radius=1.5, n_samples=20, recovery_tol=0.15)
    r = c.post(f"/api/project/{pid}/ims_analysis", json=payload3)
    assert r.status_code == 200
    ims = r.get_json()
    assert len(ims["manifold"]["points"]) == 30
    assert ims["recoverability"]["index"] == 1.0


def test_converter_topology_projects_recover_with_default_settings():
    """
    Every converter_topology project must actually recover with its OWN
    registered defaults -- this is the specific check that would have
    caught both bugs found during development (too-short horizon,
    too-coarse manifold sweep) if it had existed first.
    """
    c = _client()
    for pid in ("buck_converter", "boost_converter", "buckboost_converter"):
        project = PROJECTS[pid]
        payload = {"params": {}, "nominal_input": project["default_input"]}

        r = c.post(f"/api/project/{pid}/equilibrium", json=payload)
        assert r.status_code == 200, f"{pid}: {r.get_json()}"
        eq = r.get_json()
        assert eq["stable"] is True

        payload2 = dict(
            payload, x_star=eq["x_star"],
            disturbance={"type": project["disturbance_type"], "value": project["default_disturbance"]},
            horizon=project["default_horizon"],
            sweep_range=project["sweep_range"], sweep_points=project["sweep_points"],
            radius=project["default_radius"], n_samples=project["default_samples"],
            recovery_tol=project["default_tol"],
        )
        r = c.post(f"/api/project/{pid}/ims_analysis", json=payload2)
        assert r.status_code == 200, f"{pid}: {r.get_json()}"
        rec = r.get_json()["recoverability"]
        assert rec["index"] > 0.9, f"{pid}: recoverability index only {rec['index']} with registered defaults"


def test_buck_converter_matches_textbook_ratio():
    c = _client()
    payload = {"params": {}, "nominal_input": [0.4]}
    r = c.post("/api/project/buck_converter/equilibrium", json=payload)
    eq = r.get_json()
    expected = 0.4 * 12.0
    assert abs(eq["x_star"][0] - expected) / expected < 0.02  # small R_L drop from ideal


def test_network_mrc_full_staged_pipeline():
    c = _client()
    pid = "network_auto_mrc"

    r = c.post(f"/api/project/{pid}/build", json={})
    assert r.status_code == 200
    build = r.get_json()
    for sym in ("R", "L", "C", "P", "k_m"):
        assert sym in build["control_law_str"]

    r = c.post(f"/api/project/{pid}/equilibrium", json={})
    eq = r.get_json()
    assert eq["x_star"] == [400.0, 25.0, 405.0]

    r = c.post(f"/api/project/{pid}/simulate", json={"x_star": eq["x_star"]})
    sim = r.get_json()
    assert sim["controller_info"]["type"] == "IMS-Native MRC (Symbolic Engine derived)"
    km = sim["km"]
    em0 = sim["residual"][0]
    t = sim["trajectory_open_loop"]["t"]
    idx = min(range(len(t)), key=lambda i: abs(t[i] - 0.01))
    predicted = em0 * np.exp(-km * t[idx])
    assert abs(sim["residual"][idx] - predicted) < 0.05 * abs(em0)

    r = c.post(f"/api/project/{pid}/ims_analysis", json={"x_star": eq["x_star"]})
    assert r.status_code == 200
    assert "note" in r.get_json()


def test_all_projects_have_a_working_build_stage():
    """Cheap smoke test: every registered project's build stage must succeed with pure defaults."""
    c = _client()
    for pid, project in PROJECTS.items():
        payload = {"params": {}}
        if project["category"] != "network_mrc":
            payload["nominal_input"] = project["default_input"]
        r = c.post(f"/api/project/{pid}/build", json=payload)
        assert r.status_code == 200, f"{pid}: {r.get_json()}"


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
