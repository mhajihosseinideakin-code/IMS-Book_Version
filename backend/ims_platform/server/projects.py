"""
server.projects
-----------------

The built-in project registry: a single, backend-side source of truth
for the example projects the IMS Platform Explorer's landing page lists.
The frontend never hard-codes a project list; it fetches this registry
via `/api/projects` and renders whatever it contains, so adding a new
built-in project is a one-entry change here, not a frontend change.

Each project declares which backend "kind" of analysis it runs
(`single_model`, `converter_topology`, or `network_mrc_demo` -- the
three request shapes `server.app` implements) and a `default_payload`
matching that endpoint's expected request body, so the frontend can open
a project and immediately have a fully-populated, ready-to-run form.
"""

from __future__ import annotations

PROJECTS = [
    {
        "id": "grid_forming_inverter",
        "name": "Grid-Forming Inverter",
        "category": "Single Device",
        "description": "Droop-controlled inverter, swing-equation-type dynamics. A smoothly damped nonlinear oscillator with a large basin of attraction.",
        "kind": "single_model",
        "default_payload": {
            "model": "grid_forming_inverter",
            "state_guess": [0.3, 0.0],
            "nominal_input": [0.5],
            "sweep_range": [0.0, 0.9],
            "sweep_points": 60,
            "disturbance": {"type": "offset", "value": [1.0, 2.0]},
            "horizon": 10.0,
            "recoverability_enabled": True,
            "radius": 2.0,
            "n_samples": 100,
            "recovery_tol": 0.15,
            "mrc_enabled": True,
            "Q_diag": [8.0, 2.0],
            "R_diag": [0.05],
            "u_min": [-0.2],
            "u_max": [1.2],
        },
    },
    {
        "id": "dc_microgrid_cpl",
        "name": "DC Microgrid (Constant-Power Load)",
        "category": "Single Device",
        "description": "The classical Middlebrook large-signal instability: a genuine stable/saddle equilibrium pair, giving recoverability assessment a real, non-arbitrary boundary to detect.",
        "kind": "single_model",
        "default_payload": {
            "model": "dc_microgrid_cpl",
            "state_guess": [0.44, 1.13],
            "nominal_input": [0.5],
            "sweep_range": [0.05, 1.0],
            "sweep_points": 60,
            "disturbance": {"type": "absolute", "value": [0.68894065, 0.29546767]},
            "horizon": 3.0,
            "recoverability_enabled": True,
            "radius": 0.9,
            "n_samples": 100,
            "recovery_tol": 0.1,
            "mrc_enabled": True,
            "Q_diag": [1.0, 25.0],
            "R_diag": [0.5],
            "u_min": [0.05],
            "u_max": [1.5],
        },
    },
    {
        "id": "buck_converter",
        "name": "Buck Converter",
        "category": "Converter Topology",
        "description": "Averaged step-down converter. Ideal steady-state ratio V_o/V_in = D, checked against the platform's own equilibrium solver.",
        "kind": "converter_topology",
        "default_payload": {
            "topology": "buck", "v_in": 12.0, "L": 0.001, "R_L": 0.05, "C": 0.02, "R_load": 5.0, "duty": 0.4,
            "recoverability_enabled": True, "radius": 1.0, "n_samples": 60, "recovery_tol": 0.05,
        },
    },
    {
        "id": "boost_converter",
        "name": "Boost Converter",
        "category": "Converter Topology",
        "description": "Averaged step-up converter. Ideal steady-state ratio V_o/V_in = 1/(1-D); accuracy is sensitive to winding resistance at high duty ratio, a real property of the topology.",
        "kind": "converter_topology",
        "default_payload": {
            "topology": "boost", "v_in": 12.0, "L": 0.001, "R_L": 0.05, "C": 0.02, "R_load": 20.0, "duty": 0.5,
            "recoverability_enabled": True, "radius": 1.0, "n_samples": 60, "recovery_tol": 0.05,
        },
    },
    {
        "id": "buckboost_converter",
        "name": "Buck-Boost Converter",
        "category": "Converter Topology",
        "description": "Averaged inverting buck-boost converter (output-magnitude convention). Ideal steady-state ratio V_o/V_in = D/(1-D).",
        "kind": "converter_topology",
        "default_payload": {
            "topology": "buckboost", "v_in": 12.0, "L": 0.001, "R_L": 0.05, "C": 0.02, "R_load": 10.0, "duty": 0.4,
            "recoverability_enabled": True, "radius": 1.0, "n_samples": 60, "recovery_tol": 0.05,
        },
    },
    {
        "id": "network_auto_mrc",
        "name": "Network + Auto-MRC Demo",
        "category": "Network",
        "description": "A network assembled from primitives, driven by a Manifold-Reshaping Control law derived automatically by the Symbolic Engine -- no linearisation, no Riccati equation, no hand tuning.",
        "kind": "network_mrc_demo",
        "default_payload": {
            "R": 0.20, "L": 1.5e-3, "C": 2.5e-3, "P": 10e3, "km": 500.0,
            "v_bus_init": 400.0, "disturbance": [-5.0, 0.0, 0.0], "horizon": 0.02,
        },
    },
]

PROJECTS_BY_ID = {p["id"]: p for p in PROJECTS}
