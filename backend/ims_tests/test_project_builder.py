"""
Tests for the Project Builder's network-spec-to-Network assembler
(server.app._build_network_from_spec). Validation strategy, matching
every other numerical addition in this project:

1. Recreate an ALREADY-VALIDATED example project's exact topology from
   a spec and confirm the equilibrium matches to machine precision --
   the strongest available check for the common (single-input) case.
2. Build a genuinely NEW topology (never hardcoded anywhere) and check
   it against independent physics (KCL at every bus), not just "did it
   converge".
3. A dedicated regression test for a real bug this validation process
   found: Converter and ConstantPowerLoad both declare has_input=True,
   so a network with both (in the wrong order) silently routes the
   exogenous input to the wrong component -- self-consistent (KCL still
   balances, since the solver just finds a different but still-real
   equilibrium) but a materially wrong operating point. Confirmed by
   reproducing it against the OLD (pre-fix) order-dependent resolution
   before the fix made it a required, explicit choice instead.
"""
import numpy as np
import pytest

from ims_platform.server.app import _build_network_from_spec
from ims_platform.network import AutomaticModelBuilder, BoostModel, ConstantPowerLoad


def test_spec_builder_recreates_buck_converter_exactly():
    spec = {
        "buses": [{"id": "out", "type": "dynamic", "C": 0.02, "v_init": 4.8}],
        "converters": [{"id": "conv", "bus": "out", "topology": "buck",
                         "params": {"v_in": 12.0, "L": 1e-3, "R_L": 0.05},
                         "controller": "constant_duty", "controller_params": {"d": 0.4}}],
        "loads": [{"id": "load", "bus": "out", "type": "impedance", "R": 5.0}],
    }
    net, input_id = _build_network_from_spec(spec)
    assert input_id == "conv"
    system = AutomaticModelBuilder.build(net, input_component_id=input_id)
    eq = system.find_equilibrium(np.array([4.8, 0.5]), u=np.array([0.4]))
    known_x_star = [4.752475247524753, 0.9504950495049506]
    assert np.allclose(eq.x_star, known_x_star, atol=1e-6)


def test_spec_builder_new_two_bus_topology_satisfies_kcl_at_every_bus():
    """
    A genuinely new topology: boost converter -> bus1 -> R-L line ->
    bus2 -> CPL. Checked against real KCL (independently computed from
    the component objects' own current_injection/port_current methods),
    not just "the solver reported success".
    """
    spec = {
        "buses": [
            {"id": "bus1", "type": "dynamic", "C": 0.01, "v_init": 48.0},
            {"id": "bus2", "type": "dynamic", "C": 0.01, "v_init": 46.0},
        ],
        "lines": [{"id": "line1", "from_bus": "bus1", "to_bus": "bus2", "R": 0.1, "L": 1e-4}],
        "converters": [{"id": "conv1", "bus": "bus1", "topology": "boost",
                         "params": {"v_in": 24.0, "L": 2e-3, "R_L": 0.02},
                         "controller": "constant_duty", "controller_params": {"d": 0.5}}],
        "loads": [{"id": "cpl1", "bus": "bus2", "type": "cpl", "P": 200.0}],
        "input_component_id": "conv1",
    }
    net, input_id = _build_network_from_spec(spec)
    system = AutomaticModelBuilder.build(net, input_component_id=input_id)
    eq = system.find_equilibrium(np.array([48.5, 48.0, 4.2, 4.2]), u=np.array([0.5]))
    assert eq.residual_norm < 1e-6

    v1, v2, i_line, i_L = eq.x_star
    boost = BoostModel(v_in=24.0, L=2e-3, R_L=0.02)
    i_port_conv = boost.port_current(v1, np.array([i_L]), 0.5, boost.params)
    cpl = ConstantPowerLoad(id="cpl1", bus="bus2", P=200.0)
    i_cpl_inj = cpl.current_injection(v2, np.zeros(0), None)

    assert abs(i_port_conv - i_line) < 1e-6          # KCL at bus1 (converter's own bus)
    assert abs(i_line + i_cpl_inj) < 1e-5            # KCL at bus2 (arriving line current + injected load current = 0)


def test_ambiguous_input_component_requires_explicit_choice():
    """
    Regression test for the real bug found during validation: a network
    with both a Converter and a ConstantPowerLoad (both has_input=True)
    must require an explicit input_component_id rather than silently
    picking whichever was added first.
    """
    spec = {
        "buses": [
            {"id": "bus1", "type": "dynamic", "C": 0.01, "v_init": 48.0},
            {"id": "bus2", "type": "dynamic", "C": 0.01, "v_init": 46.0},
        ],
        "lines": [{"id": "line1", "from_bus": "bus1", "to_bus": "bus2", "R": 0.1, "L": 1e-4}],
        "converters": [{"id": "conv1", "bus": "bus1", "topology": "boost",
                         "params": {"v_in": 24.0, "L": 2e-3, "R_L": 0.02},
                         "controller": "constant_duty", "controller_params": {"d": 0.5}}],
        "loads": [{"id": "cpl1", "bus": "bus2", "type": "cpl", "P": 200.0}],
        # deliberately no input_component_id
    }
    try:
        _build_network_from_spec(spec)
        assert False, "expected a ValueError for the ambiguous input component"
    except ValueError as e:
        assert "conv1" in str(e) and "cpl1" in str(e)


def test_ambiguous_input_silently_wrong_without_the_fix():
    """
    Directly demonstrates the bug this validation caught: Network's own
    find_input_component() (unpatched, order-dependent) picks the FIRST
    has_input component -- which is the CPL here, since loads are
    conventionally added before converters -- silently routing the
    exogenous input to override the CPL's power instead of the
    converter's duty ratio. This is why the fix in
    _build_network_from_spec makes the choice explicit and required
    instead of relying on this ordering.
    """
    from ims_platform.network import Network, Bus, Line, Converter, ConstantDutyController

    net = Network()
    net.add_bus(Bus(id="bus1", C=0.01, v_init=48.0))
    net.add_bus(Bus(id="bus2", C=0.01, v_init=46.0))
    net.add_line(Line(id="line1", from_bus="bus1", to_bus="bus2", R=0.1, L=1e-4))
    net.add_component(ConstantPowerLoad(id="cpl1", bus="bus2", P=200.0))  # added first
    net.add_component(Converter(id="conv1", bus="bus1",
                                 electrical_model=BoostModel(v_in=24.0, L=2e-3, R_L=0.02),
                                 controller=ConstantDutyController(d=0.5)))
    picked = net.find_input_component()
    assert picked.id == "cpl1", "confirms the ordering ambiguity this fix guards against is real"


def _two_bus_boost_spec():
    return {
        "name": "Two-bus boost demo",
        "buses": [
            {"id": "bus1", "type": "dynamic", "C": 0.01, "v_init": 48.0},
            {"id": "bus2", "type": "dynamic", "C": 0.01, "v_init": 46.0},
        ],
        "lines": [{"id": "line1", "from_bus": "bus1", "to_bus": "bus2", "R": 0.1, "L": 1e-4}],
        "converters": [{"id": "conv1", "bus": "bus1", "topology": "boost",
                         "params": {"v_in": 24.0, "L": 2e-3, "R_L": 0.02},
                         "controller": "constant_duty", "controller_params": {"d": 0.5}}],
        "loads": [{"id": "cpl1", "bus": "bus2", "type": "cpl", "P": 200.0}],
        "input_component_id": "conv1",
    }


def _client():
    from ims_platform.server import create_app
    return create_app().test_client()


def test_custom_project_build_endpoint():
    c = _client()
    r = c.post("/api/custom_project/build", json={"network_spec": _two_bus_boost_spec(), "nominal_input": [0.5]})
    assert r.status_code == 200
    d = r.get_json()
    assert d["state_names"] == ["v_bus1", "v_bus2", "i_line1", "conv1_em_i_L"]
    assert d["network_summary"]["converter_topologies"] == {"BoostModel": 1}
    assert d["controller_description"] == "conv1: ConstantDutyController"


def test_custom_project_equilibrium_endpoint_matches_standalone_validation():
    c = _client()
    r = c.post("/api/custom_project/equilibrium", json={
        "network_spec": _two_bus_boost_spec(), "nominal_input": [0.5],
        "state_guess": [48.5, 48.0, 4.2, 4.2],
    })
    assert r.status_code == 200
    d = r.get_json()
    known = [47.66128893866421, 47.23790011199447, 4.2338882666973925, 8.467776533394785]
    assert np.allclose(d["x_star"], known, atol=1e-6)


def test_custom_project_simulate_endpoint():
    c = _client()
    x_star = [47.66128893866421, 47.23790011199447, 4.2338882666973925, 8.467776533394785]
    r = c.post("/api/custom_project/simulate", json={
        "network_spec": _two_bus_boost_spec(), "nominal_input": [0.5], "x_star": x_star,
        "disturbance": {"type": "offset", "value": [1.0, -1.0, 0.0, 0.0]}, "horizon": 0.02,
    })
    assert r.status_code == 200
    assert r.get_json()["trajectory_open_loop"]["success"] is True


def test_custom_project_ims_analysis_endpoint_and_boundary_tracing_correctly_declines_for_4_states():
    c = _client()
    x_star = [47.66128893866421, 47.23790011199447, 4.2338882666973925, 8.467776533394785]
    r = c.post("/api/custom_project/ims_analysis", json={
        "network_spec": _two_bus_boost_spec(), "nominal_input": [0.5], "x_star": x_star,
        "sweep_range": [0.3, 0.7], "sweep_points": 20, "radius": 2.0, "n_samples": 10, "recovery_tol": 0.5,
        "trace_boundary": True,
    })
    assert r.status_code == 200
    d = r.get_json()
    assert len(d["manifold"]["points"]) == 20
    assert "recoverability_boundary" not in d  # 4-state system, correctly declined rather than faking a slice


def test_custom_project_endpoint_rejects_missing_network_spec():
    c = _client()
    r = c.post("/api/custom_project/build", json={"nominal_input": [0.5]})
    assert r.status_code == 400


def test_custom_project_endpoint_surfaces_unsupported_component_errors_clearly():
    c = _client()
    spec = {
        "buses": [{"id": "b1", "type": "dynamic", "C": 0.01, "v_init": 48.0}],
        "loads": [{"id": "zip1", "bus": "b1", "type": "zip"}],  # not implemented
    }
    r = c.post("/api/custom_project/build", json={"network_spec": spec, "nominal_input": [0.5]})
    assert r.status_code == 400
    assert "zip" in r.get_json()["error"].lower() or "unsupported" in r.get_json()["error"].lower()


def test_ui_generated_spec_produces_correct_physics():
    """
    Feeds the EXACT spec produced by tests/test_project_builder_ui.js's
    simulated user interactions (add a bus, a buck converter with
    R_L=0.02 -- the builder form's hardcoded default, distinct from
    buck_converter's own R_L=0.05 -- and an impedance load) into the
    real backend, and confirms the resulting equilibrium matches an
    independent fixed-point iteration of the same physics computed by
    hand, not just "did it return 200".
    """
    import json
    import os
    spec_path = "/tmp/builder_ui_test_spec.json"
    if not os.path.exists(spec_path):
        import subprocess
        subprocess.run(["node", os.path.join(os.path.dirname(__file__), "test_project_builder_ui.js")],
                        check=True, capture_output=True)
    spec = json.load(open(spec_path))

    c = _client()
    r = c.post("/api/custom_project/equilibrium", json={
        "network_spec": spec, "nominal_input": [0.4], "state_guess": [4.8, 0.5],
    })
    assert r.status_code == 200
    x_star = r.get_json()["x_star"]

    # Independent hand-computed fixed point: v = d*v_in - R_L*i, i = v/R_load
    v, i = 4.8, 0.5
    for _ in range(200):
        i = v / 5.0
        v = 0.4 * 12.0 - 0.02 * i
    assert abs(x_star[0] - v) < 1e-6
    assert abs(x_star[1] - i) < 1e-6


def test_sanitize_for_json_replaces_nan_and_infinity():
    from ims_platform.server.app import _sanitize_for_json
    import math
    nested = {
        "a": float("inf"), "b": float("-inf"), "c": float("nan"), "d": 3.5,
        "e": [float("inf"), 1.0, {"f": float("nan")}], "g": "text", "h": None, "i": True,
    }
    out = _sanitize_for_json(nested)
    assert out["a"] is None and out["b"] is None and out["c"] is None
    assert out["d"] == 3.5
    assert out["e"] == [None, 1.0, {"f": None}]
    assert out["g"] == "text" and out["h"] is None and out["i"] is True


def test_custom_equilibrium_with_extreme_inductance_returns_valid_json():
    """
    Regression test for the exact bug reported: an extreme (but
    user-enterable) converter inductance converges to a genuinely valid
    equilibrium (tiny residual, passing the existing convergence check)
    while jacobian_condition_number overflows to literal inf during the
    post-hoc SVD-based condition-number computation. Before the fix,
    Flask's default JSON encoder would emit the literal token `Infinity`
    in the response body -- valid Python, NOT valid JSON per RFC 8259 --
    which a strict browser JSON.parse() correctly rejects. This test
    parses the raw response with Python's own strict json.loads (the
    same rules a browser follows) rather than trusting Flask's test
    client's already-lenient re-parsing, so it would have caught the
    original bug.
    """
    import json as json_module

    c = _client()
    spec = {
        "buses": [{"id": "out", "type": "dynamic", "C": 0.02, "v_init": 4.8}],
        "converters": [{"id": "conv", "bus": "out", "topology": "buck",
                         "params": {"v_in": 12.0, "L": 1e-80, "R_L": 0.05},
                         "controller": "constant_duty", "controller_params": {"d": 0.4}}],
        "loads": [{"id": "load", "bus": "out", "type": "impedance", "R": 5.0}],
        "input_component_id": "conv",
    }
    r = c.post("/api/custom_project/equilibrium", json={
        "network_spec": spec, "nominal_input": [0.4], "state_guess": [4.8, 0.5],
    })
    assert r.status_code == 200
    raw = r.get_data(as_text=True)
    assert "Infinity" not in raw and "NaN" not in raw, "raw response body must never contain the literal Infinity/NaN tokens"
    parsed = json_module.loads(raw)  # strict parse -- raises if this regresses
    assert parsed["solver_diagnostics"]["jacobian_condition_number"] is None


def test_full_custom_project_pipeline_always_returns_valid_json_including_on_solver_difficulty():
    """
    Broader sweep per the user's explicit ask: create custom projects
    through the Project Builder's own spec shape and confirm every
    stage's raw response is strictly valid JSON, across both a normal
    case and several deliberately extreme/near-degenerate parameter
    choices a real user might plausibly enter.
    """
    import json as json_module

    c = _client()
    extreme_cases = [
        {"L": 0.001, "R_L": 0.05},   # normal, sanity baseline
        {"L": 1e-80, "R_L": 0.05},   # the reproduced bug case
        {"L": 1e-20, "R_L": 0.05},   # large-but-finite condition number
        {"L": 1e-10, "R_L": 1e-12},  # near-zero damping
    ]
    for case in extreme_cases:
        spec = {
            "buses": [{"id": "out", "type": "dynamic", "C": 0.02, "v_init": 4.8}],
            "converters": [{"id": "conv", "bus": "out", "topology": "buck",
                             "params": {"v_in": 12.0, "L": case["L"], "R_L": case["R_L"]},
                             "controller": "constant_duty", "controller_params": {"d": 0.4}}],
            "loads": [{"id": "load", "bus": "out", "type": "impedance", "R": 5.0}],
            "input_component_id": "conv",
        }
        r = c.post("/api/custom_project/equilibrium", json={
            "network_spec": spec, "nominal_input": [0.4], "state_guess": [4.8, 0.5],
        })
        raw = r.get_data(as_text=True)
        # Python's own json.loads is ALSO non-strict about Infinity/NaN by
        # default (a Python-specific extension, unlike RFC 8259), so it
        # would NOT have caught the original bug on its own -- check the
        # literal tokens directly, which is what a strict browser
        # JSON.parse() actually rejects.
        assert "Infinity" not in raw and "NaN" not in raw, f"case {case} leaked a non-JSON-safe literal: {raw[:300]!r}"
        try:
            json_module.loads(raw)
        except json_module.JSONDecodeError as e:
            raise AssertionError(f"case {case} produced invalid JSON: {raw[:300]!r}") from e


def test_genuinely_non_convergent_equilibrium_returns_clean_error_not_garbage():
    """
    A short-circuit load (R=0) causes a real division-by-zero inside
    the component's own physics, propagating NaN into the Jacobian and
    raising LinAlgError deep inside the solve. Confirms this surfaces as
    a clean {"error": "..."} JSON response (still strictly valid JSON),
    not a crash or a raw traceback leaking to the client.
    """
    import json as json_module

    c = _client()
    spec = {
        "buses": [{"id": "out", "type": "dynamic", "C": 0.02, "v_init": 4.8}],
        "converters": [{"id": "conv", "bus": "out", "topology": "buck",
                         "params": {"v_in": 12.0, "L": 1e-3, "R_L": 0.05},
                         "controller": "constant_duty", "controller_params": {"d": 0.4}}],
        "loads": [{"id": "load", "bus": "out", "type": "impedance", "R": 0.0}],
        "input_component_id": "conv",
    }
    r = c.post("/api/custom_project/equilibrium", json={
        "network_spec": spec, "nominal_input": [0.4], "state_guess": [4.8, 0.5],
    })
    assert r.status_code == 400
    raw = r.get_data(as_text=True)
    parsed = json_module.loads(raw)  # must still be strictly valid JSON even on failure
    assert "error" in parsed


def test_equilibrium_response_is_always_strict_json_even_when_the_solver_struggles():
    """
    Regression test for a real bug report: a PI-controlled converter
    creates a structurally marginally-stable system (its integrator
    state gives the Jacobian a genuine zero eigenvalue by construction,
    at every equilibrium, not as a rare fluke). This makes
    np.linalg.cond(J) prone to returning float('inf') directly -- which
    it does NOT raise an exception for -- and Python's json module
    permissively serialises inf/-inf/nan as the non-standard tokens
    Infinity/-Infinity/NaN, which a strict client-side JSON.parse (a
    browser) correctly rejects with "Unexpected token 'I'".

    This test uses json.dumps(..., allow_nan=False) specifically
    because that's what actually mirrors a browser's strict parser --
    Python's own json.loads would happily accept the malformed
    "Infinity" token, so testing with json.loads alone would not have
    caught this bug.
    """
    import json as json_module

    c = _client()
    spec = {
        "buses": [{"id": "out", "type": "dynamic", "C": 0.02, "v_init": 4.8}],
        "converters": [{"id": "conv", "bus": "out", "topology": "buck",
                         "params": {"v_in": 12.0, "L": 1e-3, "R_L": 0.05},
                         "controller": "pi", "controller_params": {"v_nom": 5.0, "Kp": 0.01, "Ki": 0.1}}],
        "loads": [{"id": "load", "bus": "out", "type": "impedance", "R": 5.0}],
        "input_component_id": "conv",
    }
    # Several initial guesses, since whether np.linalg.cond returns inf
    # directly or raises (both already handled, but by different code
    # paths) can depend on exact floating-point conditions.
    for guess in ([4.8, 0.5, 0.0], [5.0, 1.0, 0.0], [5.0, 1.0, 1.0], [4.8, 0.5, 5.0], [5.0, 1.0, 100.0]):
        r = c.post("/api/custom_project/equilibrium", json={
            "network_spec": spec, "nominal_input": [5.0], "state_guess": guess,
        })
        raw_text = r.get_data(as_text=True)
        try:
            json_module.loads(raw_text)
        except json_module.JSONDecodeError as e:
            raise AssertionError(f"response was not even loadable JSON for guess={guess}: {e}\n{raw_text}")
        parsed = json_module.loads(raw_text)
        try:
            json_module.dumps(parsed, allow_nan=False)
        except ValueError as e:
            raise AssertionError(f"response contained a non-strict-JSON token (Infinity/-Infinity/NaN) "
                                  f"for guess={guess}: {e}\nraw response: {raw_text}")
        # Either a clean 200 with a genuinely finite result, or a clean
        # 400 with an error message -- never a 200 with unparseable
        # numbers silently embedded in it.
        assert r.status_code in (200, 400)
        if r.status_code == 400:
            assert "error" in parsed


def test_singular_jacobian_condition_number_is_null_not_infinity():
    """
    Direct unit test of the source-level fix in
    core.system.DynamicalSystem.find_equilibrium: a Jacobian with a
    zero row (confirmed standalone to make np.linalg.cond return
    float('inf') directly, no exception) must produce
    jacobian_condition_number=None, not a literal inf that would later
    serialise as the invalid "Infinity" token.
    """
    from ims_platform.core.system import DynamicalSystem
    import numpy as np

    class DecoupledSystem(DynamicalSystem):
        state_names = ("x1", "x2")
        input_names = ("u",)

        def default_params(self):
            return {}

        def dynamics(self, t, x, u, p):
            return np.array([0.0, -5.0 * x[1]])  # x1 has zero effect on anything -- zero row in J

    sys_ = DecoupledSystem()
    eq = sys_.find_equilibrium(np.array([0.1, 0.1]), u=np.array([0.0]))
    assert eq.jacobian_condition_number is None


def test_poor_initial_guess_can_converge_to_a_spurious_cpl_collapse_equilibrium():
    """
    A more serious finding than simple divergence: for a CPL-loaded
    network, a poor initial guess (e.g. the Project Builder's old
    hardcoded default of 10.0 for every voltage state, regardless of
    the bus's actual operating voltage) doesn't just risk fsolve
    diverging -- it can converge CLEANLY (passing the residual check)
    to a genuine, self-consistent, but physically WRONG equilibrium:
    a low-voltage/high-current "collapsed" solution, the CPL's own
    well-known bistability, rather than the intended high-voltage/
    low-current operating point. Both are real roots of the same
    equations; only the initial guess determines which one Newton's
    method finds. This is why the Project Builder's default guess was
    changed to use the user's own v_init/load-power spec data instead
    of a blind constant (see tests/test_project_builder_ui.js).
    """
    c = _client()
    spec = {
        "buses": [{"id": "bus1", "type": "dynamic", "C": 1e-5, "v_init": 10000.0}],
        "converters": [{"id": "conv1", "bus": "bus1", "topology": "buck",
                         "params": {"v_in": 20000.0, "L": 0.01, "R_L": 0.5},
                         "controller": "constant_duty", "controller_params": {"d": 0.5}}],
        "loads": [{"id": "cpl1", "bus": "bus1", "type": "cpl", "P": 500000.0}],
        "input_component_id": "conv1",
    }
    r_bad_guess = c.post("/api/custom_project/equilibrium", json={
        "network_spec": spec, "nominal_input": [0.5], "state_guess": [10.0, 0.5],
    })
    r_good_guess = c.post("/api/custom_project/equilibrium", json={
        "network_spec": spec, "nominal_input": [0.5], "state_guess": [10000.0, 50.0],
    })
    assert r_bad_guess.status_code == 200 and r_good_guess.status_code == 200

    v_bad, i_bad = r_bad_guess.get_json()["x_star"]
    v_good, i_good = r_good_guess.get_json()["x_star"]

    # The "bad guess" result is a genuine, self-consistent CPL current draw
    # at its own (wrong) voltage -- confirming it's a real equilibrium,
    # not a solver glitch -- just the undesired collapsed branch.
    assert abs(i_bad - 500000.0 / v_bad) / i_bad < 1e-3
    assert v_bad < 100  # collapsed branch: far below the intended ~10kV operating point

    # The "good guess" result matches physical expectation: buck output
    # roughly v_in * d = 20000 * 0.5 = 10000 V.
    assert abs(v_good - 10000.0) < 200
    assert abs(i_good - 500000.0 / v_good) / i_good < 1e-3


def test_isolated_cpl_only_island_rejected_before_attempting_solve():
    """
    Regression test for the actual reported bug: an isolated bus with
    only a CPL attached has NO finite equilibrium (confirmed
    independently: its own KCL equation -P/v_eff(v)=0 has no root for
    any finite v > 0), and the connectivity check must catch this
    BEFORE attempting the doomed solve, rather than letting fsolve
    diverge toward float64's overflow boundary.
    """
    c = _client()
    spec = {
        "buses": [
            {"id": "bus1", "type": "dynamic", "C": 0.01, "v_init": 48.0},
            {"id": "bus2", "type": "dynamic", "C": 0.01, "v_init": 48.0},
        ],
        "converters": [{"id": "conv1", "bus": "bus1", "topology": "buck",
                         "params": {"v_in": 100.0, "L": 1e-3, "R_L": 0.05},
                         "controller": "constant_duty", "controller_params": {"d": 0.5}}],
        "loads": [{"id": "cpl2", "bus": "bus2", "type": "cpl", "P": 500.0}],
        "input_component_id": "conv1",
    }
    r = c.post("/api/custom_project/build", json={"network_spec": spec, "nominal_input": [0.5]})
    assert r.status_code == 400
    err = r.get_json()["error"]
    assert "bus2" in err and "island" in err.lower()


def test_connectivity_check_does_not_reject_legitimate_networks():
    c = _client()
    # Single bus, converter + impedance load (buck_converter recreation)
    spec1 = {
        "buses": [{"id": "out", "type": "dynamic", "C": 0.02, "v_init": 4.8}],
        "converters": [{"id": "conv", "bus": "out", "topology": "buck",
                         "params": {"v_in": 12.0, "L": 1e-3, "R_L": 0.05},
                         "controller": "constant_duty", "controller_params": {"d": 0.4}}],
        "loads": [{"id": "load", "bus": "out", "type": "impedance", "R": 5.0}],
    }
    assert c.post("/api/custom_project/build", json={"network_spec": spec1, "nominal_input": [0.4]}).status_code == 200

    # Two connected buses (a real line joins them), converter on one, CPL on the other
    spec2 = {
        "buses": [
            {"id": "bus1", "type": "dynamic", "C": 0.01, "v_init": 48.0},
            {"id": "bus2", "type": "dynamic", "C": 0.01, "v_init": 46.0},
        ],
        "lines": [{"id": "line1", "from_bus": "bus1", "to_bus": "bus2", "R": 0.1, "L": 1e-4}],
        "converters": [{"id": "conv1", "bus": "bus1", "topology": "boost",
                         "params": {"v_in": 24.0, "L": 2e-3, "R_L": 0.02},
                         "controller": "constant_duty", "controller_params": {"d": 0.5}}],
        "loads": [{"id": "cpl1", "bus": "bus2", "type": "cpl", "P": 200.0}],
        "input_component_id": "conv1",
    }
    assert c.post("/api/custom_project/build", json={"network_spec": spec2, "nominal_input": [0.5]}).status_code == 200


def test_pi_controller_equilibrium_now_solved_correctly_via_nested_solve():
    """
    This scenario used to be REJECTED by the structural check (a
    genuine zero eigenvalue from the bare integrator makes plain joint
    Newton search unreliable -- see the module docstring for
    _try_pi_regulated_equilibrium). It is now SOLVED correctly via a
    validated nested solve: fix the regulated bus voltage at v_nom
    (forced by the controller's own d(e_int)/dt=0 equation) and solve
    the converter's own electrical-model equation for the integral
    state via a well-conditioned 1D root-find, nested inside an outer
    search over the remaining states. Checked against an independent
    physical fact (v_bus1 == v_nom exactly) and a residual well below
    the acceptance threshold -- not just "returned 200".
    """
    c = _client()
    spec = {
        "buses": [{"id": "out", "type": "dynamic", "C": 0.02, "v_init": 4.8}],
        "converters": [{"id": "conv", "bus": "out", "topology": "buck",
                         "params": {"v_in": 12.0, "L": 1e-3, "R_L": 0.05},
                         "controller": "pi", "controller_params": {"v_nom": 5.0, "Kp": 0.01, "Ki": 0.1}}],
        "loads": [{"id": "load", "bus": "out", "type": "impedance", "R": 5.0}],
        "input_component_id": "conv",
    }
    r = c.post("/api/custom_project/equilibrium", json={
        "network_spec": spec, "nominal_input": [5.0], "state_guess": [4.8, 0.5, 0.0],
    })
    assert r.status_code == 200
    d = r.get_json()
    v_out, i_L, e_int = d["x_star"]
    assert abs(v_out - 5.0) < 1e-6          # regulated bus voltage == v_nom exactly
    assert d["residual_norm"] < 1e-5
    assert d["stable"] is True


def test_pi_controller_equilibrium_on_two_bus_network_matches_reported_scenario():
    """
    The exact reported network (2 buses, 1 line, 1 PI-controlled boost
    converter feeding an impedance load on the far bus) with the exact
    guess the Explorer's own defaultStateGuess() produces (a hardcoded
    0.0 for line currents) -- confirmed this specific guess actually
    fails the FIRST attempt internally and requires the retry-with-
    perturbed-guess path to succeed, so this test also exercises that
    retry, not just the easy case.
    """
    c = _client()
    spec = {
        "buses": [
            {"id": "bus1", "type": "dynamic", "C": 0.01, "v_init": 24.0},
            {"id": "bus2", "type": "dynamic", "C": 0.01, "v_init": 24.0},
        ],
        "lines": [{"id": "line1", "from_bus": "bus1", "to_bus": "bus2", "R": 0.05, "L": 1e-4}],
        "converters": [{"id": "conv1", "bus": "bus1", "topology": "boost",
                         "params": {"v_in": 12.0, "L": 1e-3, "R_L": 0.05},
                         "controller": "pi", "controller_params": {"v_nom": 24.0, "Kp": 0.01, "Ki": 0.1}}],
        "loads": [{"id": "load", "bus": "bus2", "type": "impedance", "R": 5.0}],
        "input_component_id": "conv1",
    }
    r = c.post("/api/custom_project/equilibrium", json={
        "network_spec": spec, "nominal_input": [24.0], "state_guess": [24.0, 24.0, 0.0, 2.0, 0.0],
    })
    assert r.status_code == 200
    d = r.get_json()
    v_bus1, v_bus2, i_line1, i_L, e_int = d["x_star"]
    assert abs(v_bus1 - 24.0) < 1e-6
    assert abs(i_line1 - v_bus2 / 5.0) < 1e-4  # independent KCL check: line current == load current at bus2
    assert d["residual_norm"] < 1e-5
    assert d["stable"] is True


def test_structural_rank_check_does_not_false_positive_on_extreme_but_valid_parameters():
    """
    Regression test for a real false positive found while implementing
    the structural check: an extreme-but-physically-valid inductance
    (1e-80 H -- L doesn't even appear in the equilibrium condition, only
    the transient) produces a Jacobian row with ~1e80-scale entries that
    swamped numpy's default rank tolerance and incorrectly flagged an
    otherwise nonsingular system. Fixed by row-normalising before
    checking rank; this test confirms that fix holds.
    """
    c = _client()
    spec = {
        "buses": [{"id": "out", "type": "dynamic", "C": 0.02, "v_init": 4.8}],
        "converters": [{"id": "conv", "bus": "out", "topology": "buck",
                         "params": {"v_in": 12.0, "L": 1e-80, "R_L": 0.05},
                         "controller": "constant_duty", "controller_params": {"d": 0.4}}],
        "loads": [{"id": "load", "bus": "out", "type": "impedance", "R": 5.0}],
        "input_component_id": "conv",
    }
    r = c.post("/api/custom_project/equilibrium", json={
        "network_spec": spec, "nominal_input": [0.4], "state_guess": [4.8, 0.5],
    })
    assert r.status_code == 200


def test_describe_state_labels_match_assembler_naming_convention():
    from ims_platform.server.app import _describe_state
    assert "bus" in _describe_state("v_bus1").lower() and "bus1" in _describe_state("v_bus1")
    assert "line" in _describe_state("i_line1").lower() and "line1" in _describe_state("i_line1")
    assert "controller" in _describe_state("conv1_ctrl_e_int").lower() and "conv1" in _describe_state("conv1_ctrl_e_int")
    assert "electrical-model" in _describe_state("conv1_em_i_L").lower() and "conv1" in _describe_state("conv1_em_i_L")


def test_pi_regulated_equilibrium_helper_returns_none_for_inapplicable_networks():
    """
    The nested-solve helper must decline (return None) rather than guess
    for topologies it hasn't been validated against: more than one
    PI-regulated converter, or none at all.
    """
    from ims_platform.server.app import _try_pi_regulated_equilibrium, _build_network_from_spec
    from ims_platform.network import AutomaticModelBuilder
    import numpy as np

    spec_no_pi = {
        "buses": [{"id": "out", "type": "dynamic", "C": 0.02, "v_init": 4.8}],
        "converters": [{"id": "conv", "bus": "out", "topology": "buck",
                         "params": {"v_in": 12.0, "L": 1e-3, "R_L": 0.05},
                         "controller": "constant_duty", "controller_params": {"d": 0.4}}],
        "loads": [{"id": "load", "bus": "out", "type": "impedance", "R": 5.0}],
        "input_component_id": "conv",
    }
    net, input_id = _build_network_from_spec(spec_no_pi)
    system = AutomaticModelBuilder.build(net, input_component_id=input_id)
    result = _try_pi_regulated_equilibrium(system, net, np.array([4.8, 0.5]), np.array([0.4]))
    assert result is None


def test_pi_regulated_manifold_continuation_traces_real_range_via_adapter():
    """
    Regression test for a real reported bug: IMS Analysis timed out
    after 90s for a PI-controlled custom network. Root cause: the
    manifold sweep varies u[0], which for a PI-regulated converter IS
    the effective v_nom (SimplePIVoltageController.control_signal uses
    "p['v_nom'] if u is None else u" -- u overrides the fixed
    parameter). The plain equilibrium solver struggles at nearly every
    sweep point for the same reason single-point PI equilibria did
    before _try_pi_regulated_equilibrium existed. Fixed by wrapping the
    system with _PIAwareSystemAdapter specifically for the manifold
    continuation path, so each sweep point benefits from the same
    validated nested solve. Confirmed against the actual reported
    network: at least 25 of 30 sweep points now converge (was 1 of 30
    before this fix), completing well within a few seconds (was timing
    out past 90s).
    """
    import time
    c = _client()
    spec = {
        "buses": [
            {"id": "bus1", "type": "dynamic", "C": 0.01, "v_init": 24.0},
            {"id": "bus2", "type": "dynamic", "C": 0.01, "v_init": 24.0},
        ],
        "lines": [{"id": "line1", "from_bus": "bus1", "to_bus": "bus2", "R": 0.05, "L": 1e-4}],
        "converters": [{"id": "conv1", "bus": "bus1", "topology": "boost",
                         "params": {"v_in": 12.0, "L": 1e-3, "R_L": 0.05},
                         "controller": "pi", "controller_params": {"v_nom": 24.0, "Kp": 0.01, "Ki": 0.1}}],
        "loads": [{"id": "load", "bus": "bus2", "type": "impedance", "R": 5.0}],
        "input_component_id": "conv1",
    }
    x_star = [24.0, 23.762376237623762, 4.752475247524752, 9.914524656733429, -234.7934474029847]
    t0 = time.time()
    r = c.post("/api/custom_project/ims_analysis", json={
        "network_spec": spec, "nominal_input": [24.0], "x_star": x_star,
        "sweep_range": [12.0, 36.0], "sweep_points": 30, "radius": 1.0, "n_samples": 5, "recovery_tol": 0.1,
    })
    elapsed = time.time() - t0
    assert r.status_code == 200
    assert elapsed < 30  # was timing out past 90s before this fix
    d = r.get_json()
    assert len(d["manifold"]["points"]) >= 25  # was 1 of 30 before this fix
    # The regulated bus voltage must track the swept parameter exactly --
    # confirms the adapter is using the ACTUAL effective v_nom (u[0]),
    # not the fixed controller parameter (a real, separate bug also
    # found and fixed this session).
    for pt, alpha in zip(d["manifold"]["points"], d["manifold"]["alpha"]):
        assert abs(pt[0] - alpha) < 1e-3


def test_pi_regulated_equilibrium_uses_effective_v_nom_from_u_not_fixed_param():
    """
    Regression test for a real bug found while validating the manifold
    fix above: the nested solve was reading v_nom from the controller's
    FIXED params dict, but SimplePIVoltageController's own equations use
    "p['v_nom'] if u is None else u" -- meaning u OVERRIDES the fixed
    parameter whenever provided. This was masked in earlier tests
    because u[0] always happened to equal the fixed v_nom by
    coincidence. Confirmed directly here with u[0] DELIBERATELY
    different from the fixed v_nom parameter.
    """
    from ims_platform.server.app import _try_pi_regulated_equilibrium, _build_network_from_spec
    from ims_platform.network import AutomaticModelBuilder
    import numpy as np

    spec = {
        "buses": [{"id": "out", "type": "dynamic", "C": 0.02, "v_init": 20.0}],
        "converters": [{"id": "conv", "bus": "out", "topology": "boost",
                         "params": {"v_in": 12.0, "L": 1e-3, "R_L": 0.05},
                         "controller": "pi", "controller_params": {"v_nom": 999.0, "Kp": 0.01, "Ki": 0.1}}],
        "loads": [{"id": "load", "bus": "out", "type": "impedance", "R": 5.0}],
        "input_component_id": "conv",
    }
    net, input_id = _build_network_from_spec(spec)
    system = AutomaticModelBuilder.build(net, input_component_id=input_id)
    # u[0]=20, deliberately far from the fixed v_nom=999 in the spec --
    # the effective v_nom used must be 20 (from u), not 999.
    result = _try_pi_regulated_equilibrium(system, net, np.array([20.0, 4.0, 0.0]), np.array([20.0]))
    assert result is not None
    assert abs(result[0] - 20.0) < 1e-6


def test_deterministic_recoverability_classifies_known_contrasting_cases():
    """
    Regression test for the deterministic (non-Monte-Carlo)
    recoverability assessment, validated against the network_mrc
    project's own paper-matching manifold residual em = v_bus -
    (v_o - R*i_l) (the papers' own eq. 6): with the manifold-reshaping
    contraction gain active (km=500), the residual decays from -5.0 to
    ~-6.6e-6 in one horizon -- must classify Recoverable. With the gain
    reduced to near zero (km=0.001, mirroring the papers' own
    "MRC-no-IMS" ablation, Fig. 4), the residual barely moves (stays
    near -5.0) -- must classify Non-Recoverable.
    """
    c = _client()
    r_eq = c.post("/api/project/network_auto_mrc/equilibrium", json={"params": {}, "km": 500.0, "v_bus_init": 400.0})
    x_star = r_eq.get_json()["x_star"]
    r_ims = c.post("/api/project/network_auto_mrc/ims_analysis", json={
        "params": {}, "km": 500.0, "v_bus_init": 400.0, "x_star": x_star,
        "disturbance": [-5.0, 0.0, 0.0], "horizon": 0.05,
    })
    assert r_ims.status_code == 200
    det = r_ims.get_json()["recoverability_deterministic"]
    assert det["classification"] == "Recoverable"
    assert det["final_residual"] < 1e-4

    r_eq_b = c.post("/api/project/network_auto_mrc/equilibrium", json={"params": {}, "km": 0.001, "v_bus_init": 400.0})
    x_star_b = r_eq_b.get_json()["x_star"]
    r_ims_b = c.post("/api/project/network_auto_mrc/ims_analysis", json={
        "params": {}, "km": 0.001, "v_bus_init": 400.0, "x_star": x_star_b,
        "disturbance": [-5.0, 0.0, 0.0], "horizon": 0.05,
    })
    assert r_ims_b.status_code == 200
    det_b = r_ims_b.get_json()["recoverability_deterministic"]
    assert det_b["classification"] == "Non-Recoverable"
    assert det_b["final_residual"] > 4.0


def test_deterministic_recoverability_metrics_match_hand_computation():
    """Direct unit test of the metrics themselves against a synthetic, hand-verifiable trajectory."""
    from ims_platform.server.app import _deterministic_recoverability_assessment
    import numpy as np

    t = np.linspace(0, 1, 1000)
    residual = 5.0 * np.exp(-10.0 * t)  # a known exact exponential decay, rate=10
    result = _deterministic_recoverability_assessment(t, residual)
    assert abs(result["max_residual"] - 5.0) < 1e-6
    assert abs(result["final_residual"] - 5.0 * np.exp(-10.0)) < 1e-4
    assert abs(result["contraction_rate"] - 10.0) < 0.1  # matches the known exact rate
    assert result["classification"] == "Recoverable"


def test_ims_conditions_check_reflects_a_real_known_equilibrium_property():
    """
    The network_auto_mrc project's underlying reference model has a
    documented, genuine positive eigenvalue at its high-voltage
    equilibrium (see models/converter_cpl_paper.py's own docstring:
    "a structurally positive tangential mode... at the high-voltage
    equilibrium"). Confirmed directly: eigenvalues are approximately
    [-500, ~0, +25]. This test locks in that the tri-state IMS-conditions
    check correctly reports full_system_local_stability=VIOLATED (the
    +25 eigenvalue) and normal_hyperbolicity=VIOLATED (the ~0 eigenvalue
    sits on the imaginary axis, within the check's finite-difference-
    noise-aware tolerance) here -- an honest, real finding illustrating
    the papers' own central argument that manifold attractivity and
    traditional equilibrium-eigenvalue stability are different
    properties, not a bug to be silently smoothed over. This is the
    generic, structure-agnostic pathway (no k_m given), so the analytic
    transverse/reduced-dynamics claims are correctly NOT_ESTABLISHED --
    never inferred from the full-system finding above.
    """
    from ims_platform.server.app import _build_network_mrc, _check_ims_conditions
    import numpy as np

    system, _, _, _, _ = _build_network_mrc({"params": {}, "km": 500.0, "v_bus_init": 400.0})
    x_star = np.array([400.0, 25.0, 405.0])
    eigs = np.linalg.eigvals(system.jacobian(x_star, np.zeros(0)))
    assert any(e.real > 1.0 for e in eigs)  # confirms the known positive eigenvalue is genuinely present
    conditions = _check_ims_conditions(eigs, 275.5, 6.5e-6, 0.1)
    assert conditions["full_system_local_stability"] == "VIOLATED"
    assert conditions["normal_hyperbolicity"] == "VIOLATED"
    assert conditions["transverse_contraction"] == "NOT_ESTABLISHED"
    assert conditions["reduced_dynamics_stable"] == "NOT_ESTABLISHED"
    assert conditions["empirical_finite_horizon_recovery"] is True
    assert conditions["overall_status"] == "VIOLATED"


def test_ims_conditions_not_established_is_never_confused_with_violated():
    """Task item 2's central requirement: 'not demonstrated/not computed'
    must never be reported as mathematically 'violated'. Without a
    transverse_target_rate (the generic, structure-agnostic pathway's
    actual situation), the transverse/reduced-dynamics claims must come
    back NOT_ESTABLISHED even when the full-system eigenvalues themselves
    are all comfortably stable -- NOT_ESTABLISHED is a genuine third
    state, distinct from both SATISFIED and VIOLATED."""
    from ims_platform.server.app import _check_ims_conditions
    import numpy as np

    stable_eigs = np.array([-10.0 + 0j, -20.0 + 5j, -20.0 - 5j])
    conditions = _check_ims_conditions(stable_eigs, contraction_rate=5.0, final_residual=1e-4, tolerance=0.1)
    assert conditions["full_system_local_stability"] == "SATISFIED"
    assert conditions["normal_hyperbolicity"] == "SATISFIED"
    # never inferred from the full-system finding above, and never
    # reported as VIOLATED just because it was never computed:
    assert conditions["transverse_contraction"] == "NOT_ESTABLISHED"
    assert conditions["reduced_dynamics_stable"] == "NOT_ESTABLISHED"
    assert conditions["certified_regional_ims"] == "NOT_ESTABLISHED"
    # overall_status must also stay NOT_ESTABLISHED, not be upgraded to
    # SATISFIED on the strength of the full-system claim alone
    assert conditions["overall_status"] == "NOT_ESTABLISHED"


def test_ims_conditions_transverse_target_rate_enables_genuine_analytic_split():
    """When a manifold's own target contraction rate k_m IS known (the
    network_mrc pathway's situation), the transverse/reduced-dynamics
    claims become analytically computable -- SATISFIED here, not merely
    NOT_ESTABLISHED -- because the eigenvalue closest to -k_m is
    identified as transverse and the rest are genuinely tangential and
    stable."""
    from ims_platform.server.app import _check_ims_conditions
    import numpy as np

    # -500 is the transverse mode (matches k_m); -60.1 and -178.3+-436j are
    # tangential and all strictly stable -- mirrors the validated four-state benchmark.
    eigs = np.array([-500.0 + 0j, -60.0849 + 0j, -178.2909 + 436.0281j, -178.2909 - 436.0281j])
    conditions = _check_ims_conditions(eigs, contraction_rate=500.0, final_residual=0.0, tolerance=0.1,
                                       transverse_target_rate=500.0)
    assert conditions["full_system_local_stability"] == "SATISFIED"
    assert conditions["normal_hyperbolicity"] == "SATISFIED"
    assert conditions["transverse_contraction"] == "SATISFIED"
    assert conditions["reduced_dynamics_stable"] == "SATISFIED"
    assert conditions["overall_status"] == "SATISFIED"
    assert conditions["transverse_eigenvalue"]["re"] == pytest.approx(-500.0)
    assert len(conditions["tangential_eigenvalues"]) == 3


def test_ims_conditions_transverse_target_rate_can_reveal_genuine_violation():
    """A k_m that does NOT match any actual eigenvalue means the closed-
    loop system is not actually contracting transversally at the demanded
    rate -- this must be reported as VIOLATED, not silently accepted."""
    from ims_platform.server.app import _check_ims_conditions
    import numpy as np

    eigs = np.array([-10.0 + 0j, -20.0 + 0j, -30.0 + 0j])
    conditions = _check_ims_conditions(eigs, contraction_rate=5.0, final_residual=1e-4, tolerance=0.1,
                                       transverse_target_rate=500.0)
    assert conditions["transverse_contraction"] == "VIOLATED"
    assert conditions["overall_status"] == "VIOLATED"


def test_basic_mode_ims_analysis_is_fast_and_skips_monte_carlo():
    """
    Regression test for a real reported issue: IMS Analysis was timing
    out even in "Basic" mode because the frontend was still sending
    recoverability_enabled=true and trace_boundary unconditionally,
    regardless of the Basic/Advanced toggle -- and separately, the
    deterministic (manifold-residual) assessment never actually ran for
    this category because no disturbance was ever included in the
    ims_analysis payload at all. Confirms both are now fixed at the
    backend level: with recoverability_enabled=False and a disturbance
    included, this completes quickly and returns the deterministic
    assessment without running Monte Carlo.
    """
    import time
    c = _client()
    spec = {
        "buses": [
            {"id": "bus1", "type": "dynamic", "C": 0.01, "v_init": 24.0},
            {"id": "bus2", "type": "dynamic", "C": 0.01, "v_init": 24.0},
        ],
        "lines": [{"id": "line1", "from_bus": "bus1", "to_bus": "bus2", "R": 0.05, "L": 1e-4}],
        "converters": [{"id": "conv1", "bus": "bus1", "topology": "boost",
                         "params": {"v_in": 12.0, "L": 1e-3, "R_L": 0.05},
                         "controller": "pi", "controller_params": {"v_nom": 24.0, "Kp": 0.01, "Ki": 0.1}}],
        "loads": [{"id": "load", "bus": "bus2", "type": "impedance", "R": 5.0}],
        "input_component_id": "conv1",
    }
    r_eq = c.post("/api/custom_project/equilibrium", json={
        "network_spec": spec, "nominal_input": [24.0], "state_guess": [24.0, 24.0, 0.0, 2.0, 0.0],
    })
    x_star = r_eq.get_json()["x_star"]

    t0 = time.time()
    r = c.post("/api/custom_project/ims_analysis", json={
        "network_spec": spec, "nominal_input": [24.0], "x_star": x_star,
        "sweep_range": [12.0, 36.0], "sweep_points": 30,
        "disturbance": {"type": "offset", "value": [0.1, 0.1, 0.0, 0.0, 0.0]}, "horizon": 0.05,
        "recoverability_enabled": False, "trace_boundary": False,
    })
    elapsed = time.time() - t0
    assert r.status_code == 200
    assert elapsed < 10  # was timing out past 90s before this fix
    d = r.get_json()
    assert "recoverability_deterministic" in d
    assert "recoverability" not in d  # Monte Carlo correctly skipped


def test_limiting_state_identified_when_non_recoverable():
    """
    Regression test for the review's own stated 'most important
    scientific issue': the platform was reporting a bare
    'Non-Recoverable' classification with no explanation of which part
    of the system caused it. Confirms the response now names a specific
    state and location, with a value that's actually the largest
    per-state deviation (not an arbitrary pick).
    """
    c = _client()
    spec = {
        "buses": [{"id": f"bus{i}", "type": "dynamic", "C": 0.001, "v_init": 100.0} for i in range(1, 10)],
        "lines": [
            {"id": "line1", "from_bus": "bus1", "to_bus": "bus2", "R": 0.1, "L": 1e-4},
            {"id": "line2", "from_bus": "bus1", "to_bus": "bus3", "R": 0.1, "L": 1e-4},
            {"id": "line3", "from_bus": "bus1", "to_bus": "bus4", "R": 0.1, "L": 1e-4},
            {"id": "line4", "from_bus": "bus4", "to_bus": "bus5", "R": 0.1, "L": 1e-4},
            {"id": "line5", "from_bus": "bus5", "to_bus": "bus6", "R": 0.1, "L": 1e-4},
            {"id": "line6", "from_bus": "bus6", "to_bus": "bus7", "R": 0.1, "L": 1e-4},
            {"id": "line7", "from_bus": "bus7", "to_bus": "bus8", "R": 0.1, "L": 1e-4},
            {"id": "line8", "from_bus": "bus8", "to_bus": "bus9", "R": 0.1, "L": 1e-4},
        ],
        "converters": [
            {"id": "conv1", "bus": "bus1", "topology": "boost", "params": {"v_in": 24.0, "L": 1e-3, "R_L": 0.02}, "controller": "constant_duty", "controller_params": {"d": 0.5}},
            {"id": "conv2", "bus": "bus3", "topology": "boost", "params": {"v_in": 24.0, "L": 1e-3, "R_L": 0.02}, "controller": "constant_duty", "controller_params": {"d": 0.5}},
            {"id": "conv4", "bus": "bus8", "topology": "buck", "params": {"v_in": 24.0, "L": 1e-3, "R_L": 0.02}, "controller": "constant_duty", "controller_params": {"d": 0.5}},
        ],
        "sources": [
            {"id": "src1", "bus": "bus2", "v_source": 400.0, "R_source": 0.01},
            {"id": "src2", "bus": "bus4", "v_source": 400.0, "R_source": 0.01},
        ],
        "loads": [
            {"id": "load1", "bus": "bus9", "type": "cpl", "P": 500.0},
            {"id": "load2", "bus": "bus7", "type": "cpl", "P": 500.0},
            {"id": "load3", "bus": "bus5", "type": "cpl", "P": 500.0},
        ],
        "input_component_id": "conv1",
    }
    guess = [100.0] * 9 + [0.0] * 8 + [10.0, 10.0, 10.0]
    r_eq = c.post("/api/custom_project/equilibrium", json={"network_spec": spec, "nominal_input": [0.5], "state_guess": guess})
    x_star = r_eq.get_json()["x_star"]
    disturbed = [x + 5.0 for x in x_star]
    r_ims = c.post("/api/custom_project/ims_analysis", json={
        "network_spec": spec, "nominal_input": [0.5], "x_star": x_star,
        "sweep_range": [0.4, 0.6], "sweep_points": 10,
        "disturbance": {"type": "absolute", "value": disturbed}, "horizon": 0.05,
        "recoverability_enabled": False, "trace_boundary": False,
    })
    assert r_ims.status_code == 200
    det = r_ims.get_json()["recoverability_deterministic"]
    # This scenario now classifies as "Marginal" rather than the
    # "Non-Recoverable" it produced before the manifold's default
    # projection method changed from nearest_sample to polyline (see
    # ims/manifold.py) -- expected and correct: polyline gives a more
    # accurate (smaller) residual, and this test's actual purpose is
    # validating the limiting-state feature, not this specific
    # network's exact classification threshold.
    assert det["classification"] in ("Non-Recoverable", "Marginal")
    assert "limiting_state" in det and det["limiting_state"] in spec["converters"][0].get("params", {}).keys() or "limiting_state" in det
    assert det["limiting_state_location"]
    assert "note" in det and det["limiting_state"] in det["note"]
    # Independently confirm this really IS the largest deviation, not
    # an arbitrary field -- direct unit test of the underlying function.
    from ims_platform.server.app import _identify_limiting_state, _build_network_from_spec
    from ims_platform.network import AutomaticModelBuilder
    from ims_platform.ims import IntrinsicManifold
    import numpy as np
    net, input_id = _build_network_from_spec(spec)
    system = AutomaticModelBuilder.build(net, input_component_id=input_id)
    M = IntrinsicManifold(system, param_name="custom_param").build(
        alpha_range=(0.4, 0.6), n_points=10, x0_guess=np.array(x_star), keep_unstable=True,
    )
    x_final = np.array(disturbed)
    result = _identify_limiting_state(M, x_final, list(system.state_names))
    # Uses M.project() -- the same interface _identify_limiting_state
    # calls internally -- not the legacy M.nearest_point(), since M's
    # default projection_method is now "polyline" and the two methods
    # can identify different nearest points.
    projection = M.project(x_final)
    diffs = np.abs(x_final - np.asarray(projection.x_projected))
    expected_idx = int(np.argmax(diffs))
    assert result["limiting_state"] == system.state_names[expected_idx]


def test_geometric_recoverability_narrative_describes_actual_physics_correctly():
    """
    Regression test for a real bug found while writing this narrative:
    the first version described any fitted contraction rate above a
    tiny floor (1e-6) as "contracted... at a rate of X", which for a
    genuinely non-recoverable case (rate=0.001 s^-1, residual
    essentially unchanged: 5.0 -> 4.9997) produced a misleading claim
    of contraction happening. Fixed by also requiring the residual to
    have actually dropped substantially, not just a technically
    positive fitted rate. Validated against both known contrast cases.
    """
    from ims_platform.server.app import _geometric_recoverability_narrative

    recoverable_det = {
        "classification": "Recoverable", "max_residual": 5.0, "final_residual": 6.56e-6,
        "contraction_rate": 275.5, "recoverability_margin": 0.09999, "recoverability_margin_ratio": 0.9999,
        "tolerance_used": 0.1, "recovery_time": 0.0079,
    }
    non_recoverable_det = {
        "classification": "Non-Recoverable", "max_residual": 5.0, "final_residual": 4.9997,
        "contraction_rate": 0.001, "recoverability_margin": -4.8997, "recoverability_margin_ratio": -48.997,
        "tolerance_used": 0.1, "recovery_time": None,
    }
    rec_text = _geometric_recoverability_narrative(recoverable_det)
    assert "contracted toward the intrinsic manifold at an estimated rate of 276" in rec_text
    assert "positive recoverability margin" in rec_text

    non_rec_text = _geometric_recoverability_narrative(non_recoverable_det)
    assert "did not show meaningful contraction" in non_rec_text
    assert "contracted toward the intrinsic manifold at an estimated rate of 0.001" not in non_rec_text  # the actual bug found
    assert "negative recoverability margin" in non_rec_text


def test_recoverability_region_summary_always_present_and_honest_about_boundary_tracing():
    """
    Regression test for a real gap directly identified: Recoverability
    Region and Critical Boundary content previously only appeared when
    full directional boundary tracing had been explicitly run (Advanced
    mode) -- meaning the faster, default Basic-mode path (most reports)
    never showed these IMS concepts at all. Confirms the summary is now
    always generated and is honest about the difference between two
    distinct things: this trajectory's own direct region-membership
    answer (always available) versus a full state-space boundary
    projection (only when explicitly traced).
    """
    from ims_platform.server.app import _recoverability_region_summary

    recoverable_det = {"final_residual": 6.56e-6, "recoverability_margin": 0.09999, "recoverability_margin_ratio": 0.9999, "tolerance_used": 0.1}
    non_recoverable_det = {"final_residual": 4.9997, "recoverability_margin": -4.8997, "recoverability_margin_ratio": -48.997, "tolerance_used": 0.1}

    no_trace = _recoverability_region_summary(recoverable_det, False)
    assert no_trace["inside_region"] is True
    assert no_trace["boundary_traced"] is False
    assert "was not run" in no_trace["boundary_note"]
    assert "Advanced mode" in no_trace["boundary_note"]

    with_trace = _recoverability_region_summary(non_recoverable_det, True)
    assert with_trace["inside_region"] is False
    assert with_trace["boundary_traced"] is True
    assert "was also run" in with_trace["boundary_note"]

    # The membership explanation must correctly reflect classification
    # in BOTH directions, not just the recoverable case.
    assert "inside" in no_trace["membership_explanation"]
    assert "outside" in with_trace["membership_explanation"]


def test_manifold_residual_explanation_addresses_the_no_universal_unit_question():
    """
    Requested directly: the report should explain what the manifold
    residual represents and why it doesn't carry one universal unit,
    since the platform deliberately doesn't label it with a fixed unit.
    """
    from ims_platform.server.app import _manifold_residual_explanation
    text = _manifold_residual_explanation()
    assert "distance" in text.lower() and "intrinsic manifold" in text.lower()
    assert "unit" in text.lower()
    assert "network_mrc" in text  # names the one project where it DOES have a real unit, honestly


def test_recoverability_margin_has_correct_sign_convention():
    """
    Regression test for the recoverability margin: positive means the
    trajectory settled with room to spare inside the admissible
    tolerance (Recoverable case), negative means the tolerance was
    exceeded (Non-Recoverable case) -- validated against the same known
    contrast pair used throughout this test file (km=500 vs km=0.001).
    """
    c = _client()
    r_eq = c.post("/api/project/network_auto_mrc/equilibrium", json={"params": {}, "km": 500.0, "v_bus_init": 400.0})
    x_star = r_eq.get_json()["x_star"]
    r_ims = c.post("/api/project/network_auto_mrc/ims_analysis", json={
        "params": {}, "km": 500.0, "v_bus_init": 400.0, "x_star": x_star,
        "disturbance": [-5.0, 0.0, 0.0], "horizon": 0.05,
    })
    det = r_ims.get_json()["recoverability_deterministic"]
    assert det["recoverability_margin"] > 0
    assert det["recoverability_margin_ratio"] > 0

    r_eq2 = c.post("/api/project/network_auto_mrc/equilibrium", json={"params": {}, "km": 0.001, "v_bus_init": 400.0})
    x_star2 = r_eq2.get_json()["x_star"]
    r_ims2 = c.post("/api/project/network_auto_mrc/ims_analysis", json={
        "params": {}, "km": 0.001, "v_bus_init": 400.0, "x_star": x_star2,
        "disturbance": [-5.0, 0.0, 0.0], "horizon": 0.05,
    })
    det2 = r_ims2.get_json()["recoverability_deterministic"]
    assert det2["recoverability_margin"] < 0
    assert det2["recoverability_margin_ratio"] < 0
    # The margin must be exactly (tolerance - final_residual), not an
    # approximation -- confirm the identity directly.
    assert abs(det2["recoverability_margin"] - (det2["tolerance_used"] - det2["final_residual"])) < 1e-9


def test_resolve_projection_method_maps_auto_and_rejects_invalid():
    """
    _resolve_projection_method: "auto" maps directly to "polyline" (the
    benchmarked recommendation), explicit valid methods pass through
    unchanged, and unknown values are rejected with a clear error
    rather than silently falling back to something.
    """
    from ims_platform.server.app import _resolve_projection_method

    assert _resolve_projection_method("auto") == "polyline"
    assert _resolve_projection_method(None) == "polyline"
    assert _resolve_projection_method("") == "polyline"
    assert _resolve_projection_method("polyline") == "polyline"
    assert _resolve_projection_method("nearest_sample") == "nearest_sample"
    assert _resolve_projection_method("newton_refined") == "newton_refined"
    try:
        _resolve_projection_method("bogus")
        assert False, "invalid method should have raised ValueError"
    except ValueError as e:
        assert "bogus" in str(e)


def test_residual_projection_method_selectable_end_to_end():
    """
    Regression test for the user-selectable projection method feature:
    confirms the API actually resolves and uses each requested method
    (not just that the helper function works in isolation), that "auto"
    resolves to "polyline", that the validation summary is included for
    polyline/newton_refined but not nearest_sample, and that an invalid
    method name is rejected with HTTP 400 rather than silently falling
    back or crashing.
    """
    c = _client()
    r_eq = c.post("/api/project/buck_converter/equilibrium", json={"params": {}, "nominal_input": [0.4]})
    x_star = r_eq.get_json()["x_star"]

    base_payload = {
        "params": {}, "nominal_input": [0.4], "x_star": x_star,
        "sweep_range": [0.2, 0.6], "sweep_points": 30,
        "disturbance": {"type": "offset", "value": [0.5, 0.2]}, "horizon": 0.05,
        "recoverability_enabled": False, "trace_boundary": False,
    }

    r_auto = c.post("/api/project/buck_converter/ims_analysis", json={**base_payload, "residual_projection_method": "auto"})
    assert r_auto.status_code == 200
    d_auto = r_auto.get_json()
    assert d_auto["residual_projection_method"] == "polyline"
    assert d_auto["residual_projection_method_requested"] == "auto"
    assert "projection_method_validation_summary" in d_auto

    r_legacy = c.post("/api/project/buck_converter/ims_analysis", json={**base_payload, "residual_projection_method": "nearest_sample"})
    assert r_legacy.status_code == 200
    d_legacy = r_legacy.get_json()
    assert d_legacy["residual_projection_method"] == "nearest_sample"
    assert "projection_method_validation_summary" not in d_legacy  # no benchmark note for the legacy method itself

    r_newton = c.post("/api/project/buck_converter/ims_analysis", json={**base_payload, "residual_projection_method": "newton_refined"})
    assert r_newton.status_code == 200
    assert r_newton.get_json()["residual_projection_method"] == "newton_refined"

    r_bad = c.post("/api/project/buck_converter/ims_analysis", json={**base_payload, "residual_projection_method": "bogus"})
    assert r_bad.status_code == 400


def test_polyline_agrees_closely_with_newton_refined_at_much_lower_cost():
    """
    Locks in the benchmark comparison across all three projection
    methods requested as a final validation before trusting polyline as
    the default: on the pathological network at a resolution where
    nearest_sample still misclassifies (n=300), polyline and
    newton_refined must agree on classification exactly and on margin
    within a small tolerance, while polyline uses the same number of
    equilibrium solves as nearest_sample (build only) versus
    newton_refined's roughly 2x total (one extra solve per residual
    query). Also checks a well-behaved case (buck_converter) where all
    three methods should agree closely regardless of resolution, since
    nearest_sample's O(h) error is only large where the branch is
    genuinely steep.
    """
    from ims_platform.server.app import _build_network_from_spec, _deterministic_recoverability_assessment, PROJECTS, _build_system
    from ims_platform.network import AutomaticModelBuilder
    from ims_platform.ims import IntrinsicManifold
    from ims_platform.core import Simulator
    import numpy as np

    # Pathological case at n=300 -- the resolution where nearest_sample
    # still gives "Marginal" but the true answer (confirmed by both
    # higher-accuracy methods) is "Recoverable".
    spec = {
        "buses": [
            {"id": "bus1", "type": "dynamic", "C": 0.001, "v_init": 380.0},
            {"id": "bus2", "type": "dynamic", "C": 0.001, "v_init": 380.0},
            {"id": "bus3", "type": "dynamic", "C": 0.001, "v_init": 200.0},
            {"id": "bus4", "type": "dynamic", "C": 0.001, "v_init": 380.0},
        ],
        "lines": [
            {"id": "line1", "from_bus": "bus1", "to_bus": "bus2", "R": 0.1, "L": 1e-4},
            {"id": "line2", "from_bus": "bus2", "to_bus": "bus3", "R": 0.1, "L": 1e-4},
            {"id": "line3", "from_bus": "bus3", "to_bus": "bus4", "R": 0.1, "L": 1e-4},
        ],
        "converters": [
            {"id": "conv1", "bus": "bus3", "topology": "boost", "params": {"v_in": 24.0, "L": 1e-3, "R_L": 0.02}, "controller": "constant_duty", "controller_params": {"d": 0.5}},
        ],
        "sources": [{"id": "src1", "bus": "bus2", "v_source": 400.0, "R_source": 0.01}],
        "loads": [
            {"id": "load1", "bus": "bus4", "type": "cpl", "P": 500.0},
            {"id": "load2", "bus": "bus1", "type": "current", "I": 5.0},
        ],
        "input_component_id": "conv1",
    }
    net, input_id = _build_network_from_spec(spec)
    system = AutomaticModelBuilder.build(net, input_component_id=input_id)
    x_star = np.array([380.91556455070827, 381.41556455070827, 196.07121005779075, 195.8158681371117, -5.0, 1853.4435449291748, 2.553419206790624, -3701.7802514447685])
    disturbance = np.array([0.0, 0.0, 5.0, -5.0, 0.0, 0.0, 0.0, 0.0])
    x_disturbed = x_star + disturbance
    sim = Simulator(system, method="RK45")
    traj = sim.simulate(x_disturbed, (0.0, 0.05), u=system.default_input(), n_eval=300)

    results = {}
    for method in ["polyline", "newton_refined"]:
        M = IntrinsicManifold(system, param_name="custom_param", projection_method=method).build(
            alpha_range=(0.0, 1.0), n_points=300, x0_guess=x_star, keep_unstable=True,
        )
        residual_traj = M.residual_trajectory(traj.x)
        det = _deterministic_recoverability_assessment(traj.t, residual_traj)
        results[method] = det

    assert results["polyline"]["classification"] == results["newton_refined"]["classification"] == "Recoverable"
    margin_diff_ratio = abs(results["polyline"]["recoverability_margin"] - results["newton_refined"]["recoverability_margin"]) / abs(results["newton_refined"]["recoverability_margin"])
    assert margin_diff_ratio < 0.10  # within 10% -- benchmarked at ~4%, generous margin for solver nondeterminism

    # Well-behaved case: all three methods should agree closely regardless of resolution
    project = PROJECTS["buck_converter"]
    system2, param_name2, _ = _build_system(project, {"params": {}, "nominal_input": [0.4]})
    x_star2 = np.array([4.752475247524753, 0.9504950495049506])
    x_disturbed2 = x_star2 + np.array([0.5, 0.2])
    sim2 = Simulator(system2, method="RK45")
    traj2 = sim2.simulate(x_disturbed2, (0.0, 0.05), u=system2.default_input(), n_eval=300)
    residuals2 = {}
    for method in ["nearest_sample", "polyline", "newton_refined"]:
        M2 = IntrinsicManifold(system2, param_name=param_name2, projection_method=method).build(
            alpha_range=(0.2, 0.6), n_points=30, x0_guess=x_star2, keep_unstable=True,
        )
        residuals2[method] = _deterministic_recoverability_assessment(traj2.t, M2.residual_trajectory(traj2.x))["final_residual"]
    assert max(residuals2.values()) - min(residuals2.values()) < 0.01  # tight agreement on a well-behaved branch


def test_manifold_projection_methods_are_modular_and_correct():
    """
    Locks in the modular projection system added after a detailed
    scientific investigation into a persistently large manifold
    residual (see conversation history): nearest_sample gave a
    first-order-accurate but coarse approximation, dominated by a
    single steep state; polyline gives a second-order-accurate
    approximation using the exact same continuation samples, no extra
    equilibrium solves. Validated against the specific reconstructed
    network and numbers from that investigation, not synthetic values.
    """
    from ims_platform.server.app import _build_network_from_spec
    from ims_platform.network import AutomaticModelBuilder
    from ims_platform.ims import IntrinsicManifold
    import numpy as np

    spec = {
        "buses": [
            {"id": "bus1", "type": "dynamic", "C": 0.001, "v_init": 380.0},
            {"id": "bus2", "type": "dynamic", "C": 0.001, "v_init": 380.0},
            {"id": "bus3", "type": "dynamic", "C": 0.001, "v_init": 200.0},
            {"id": "bus4", "type": "dynamic", "C": 0.001, "v_init": 380.0},
        ],
        "lines": [
            {"id": "line1", "from_bus": "bus1", "to_bus": "bus2", "R": 0.1, "L": 1e-4},
            {"id": "line2", "from_bus": "bus2", "to_bus": "bus3", "R": 0.1, "L": 1e-4},
            {"id": "line3", "from_bus": "bus3", "to_bus": "bus4", "R": 0.1, "L": 1e-4},
        ],
        "converters": [
            {"id": "conv1", "bus": "bus3", "topology": "boost", "params": {"v_in": 24.0, "L": 1e-3, "R_L": 0.02}, "controller": "constant_duty", "controller_params": {"d": 0.5}},
        ],
        "sources": [{"id": "src1", "bus": "bus2", "v_source": 400.0, "R_source": 0.01}],
        "loads": [
            {"id": "load1", "bus": "bus4", "type": "cpl", "P": 500.0},
            {"id": "load2", "bus": "bus1", "type": "current", "I": 5.0},
        ],
        "input_component_id": "conv1",
    }
    net, input_id = _build_network_from_spec(spec)
    system = AutomaticModelBuilder.build(net, input_component_id=input_id)
    x_star = np.array([380.91556455070827, 381.41556455070827, 196.07121005779075, 195.8158681371117, -5.0, 1853.4435449291748, 2.553419206790624, -3701.7802514447685])

    # Default is polyline
    M_default = IntrinsicManifold(system, param_name="custom_param").build(alpha_range=(0.0, 1.0), n_points=30, x0_guess=x_star, keep_unstable=True)
    assert M_default.projection_method == "polyline"
    assert abs(M_default.residual(x_star) - 2.495650112434833) < 1e-6

    # Legacy explicitly selectable and gives the known first-order result
    M_legacy = IntrinsicManifold(system, param_name="custom_param", projection_method="nearest_sample").build(alpha_range=(0.0, 1.0), n_points=30, x0_guess=x_star, keep_unstable=True)
    assert abs(M_legacy.residual(x_star) - 62.19569251664009) < 1e-6

    # newton_refined now implemented (was a stub raising
    # NotImplementedError at the time this test was first written);
    # gives an even better approximation than polyline at x*, since it
    # eliminates the chord-curvature error entirely and only inherits
    # polyline's much smaller residual alpha-estimation error.
    M_refined = IntrinsicManifold(system, param_name="custom_param", projection_method="newton_refined").build(alpha_range=(0.0, 1.0), n_points=30, x0_guess=x_star, keep_unstable=True)
    r_refined = M_refined.residual(x_star)
    assert abs(r_refined - 1.1752226107808144) < 1e-4
    assert r_refined < 2.495650112434833  # strictly better than polyline
    assert r_refined < 62.19569251664009  # and dramatically better than nearest_sample

    # Invalid method name rejected explicitly, not silently accepted
    try:
        IntrinsicManifold(system, param_name="custom_param", projection_method="bogus")
        assert False, "invalid projection_method should have raised ValueError"
    except ValueError:
        pass

    # The uniform project() interface returns consistent, usable results
    proj = M_default.project(x_star)
    assert proj.method == "polyline"
    assert proj.alpha is not None and abs(proj.alpha - 0.5) < 0.05
    assert abs(proj.dist - M_default.residual(x_star)) < 1e-9


def test_jacobian_rank_and_spectral_radius_correctness():
    """
    jacobian_rank uses the same row-normalised computation validated in
    _structural_rank_check; jacobian_spectral_radius is max|eigenvalue|.
    Validated directly against the eigenvalues in the SAME response
    (not a hardcoded expectation, which would drift if the reference
    project's parameters ever change) and confirmed full rank (2/2) for
    this well-posed system.
    """
    c = _client()
    r = c.post("/api/project/buck_converter/equilibrium", json={"params": {}, "nominal_input": [0.4]})
    d = r.get_json()
    assert d["jacobian_rank"] == 2
    expected_radius = max((e["re"] ** 2 + e["im"] ** 2) ** 0.5 for e in d["eigenvalues"])
    assert abs(d["jacobian_spectral_radius"] - expected_radius) < 1e-6


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
