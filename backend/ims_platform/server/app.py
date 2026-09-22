"""
server.app
----------

Backend for the IMS Platform Explorer: a local Flask application that
serves (a) a JSON API wrapping the real, test-validated `ims_platform`
engine, staged to mirror the platform's own architecture --

    Network -> Automatic Model Builder -> Simulation -> IMS -> Report

exposed as four separate endpoints (build / equilibrium / simulate /
ims_analysis) rather than one large "run everything" call, so the
Explorer's workflow buttons each trigger a real, independent backend
step -- and (b) the Explorer's static frontend, from the same origin.

Six built-in projects are registered (PROJECTS below), covering both
architectural categories the platform currently supports:

    "single_model"        -- a hand-written DynamicalSystem (Phase V1)
    "converter_topology"  -- a Bus + ConstantImpedanceLoad + Converter
                              network, built by the Automatic Model
                              Builder (Phase V2.2-V2.4)
    "network_mrc"          -- the network + genuine, Symbolic-Engine-
                              derived MRC demo (Phase V2.5)

Run:
    python -m ims_platform.server
then open http://127.0.0.1:8765/ in a browser.
"""

from __future__ import annotations

import os
import math
import time
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, Response
from flask.json.provider import DefaultJSONProvider

from ..models import GridFormingInverter, DCMicrogridCPL, ConverterCPLPaper
from ..core import Simulator
from ..core.system import EquilibriumResult
from ..ims import IntrinsicManifold, RecoverabilityAnalyzer
from ..control import ScheduledLQRControl, MRCSynthesizer
from ..network.controller import ConstantSetpointController
from ..mrc_designer.auto_manifold import (
    run_auto_mrc_pipeline, run_auto_mrc_pipeline_multi, attach_auto_mrc, find_duty_modulated_converter,
    attach_bus_voltage_mrc, compare_baseline_m1_m2,
)
from ..mrc_designer.designer import MRCNotEstablished as _AutoMRCNotEstablished
from .auth import register_auth_routes
from . import iberian_scenario
from . import gfm_current_limit_case
from . import multi_converter_fault_case
from . import report_generator
from ..network import (
    Network, Bus, Line, ConstantPowerLoad, ConstantImpedanceLoad, ConstantCurrentLoad, IdealSource, Converter,
    BuckModel, BoostModel, BuckBoostModel, ConstantDutyController, SimplePIVoltageController, PIDutyController, DroopController,
    ConverterCPLElectricalModel, SynthesizedMRCController, AutomaticModelBuilder, NetworkValidationError,
)

def _static_dir() -> str:
    """
    Resolve the static-files directory correctly both in a normal source
    checkout and when frozen into a standalone executable by PyInstaller
    (where bundled data files live under `sys._MEIPASS`, not next to this
    source file). Deliberately duplicated (not imported) from
    `launcher.resource_base_dir` so `server.app` has no import-time
    dependency on `launcher` -- the server must keep working standalone
    (`python -m ims_platform.server`) whether or not the launcher/PyInstaller
    packaging is present at all.
    """
    import sys
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, "ims_platform", "server", "static")  # type: ignore[attr-defined]
    return os.path.join(os.path.dirname(__file__), "static")


STATIC_DIR = _static_dir()

MODEL_REGISTRY = {"grid_forming_inverter": GridFormingInverter, "dc_microgrid_cpl": DCMicrogridCPL}

CUSTOM_LOAD_TYPES = {"cpl", "impedance", "current"}
CUSTOM_CONVERTER_TOPOLOGIES = {"buck": BuckModel, "boost": BoostModel, "buckboost": BuckBoostModel}
CUSTOM_CONTROLLER_TYPES = {"constant_duty", "pi", "droop"}


def _check_network_connectivity(net: Network) -> None:
    """
    Detects a real, mathematically-grounded failure mode found while
    debugging a user's "equilibrium did not converge" report: an
    isolated bus (or island of buses, connected to each other via lines
    but not to the rest of the network) containing only loads (CPL,
    impedance, current) and no Converter or IdealSource has NO FINITE
    EQUILIBRIUM AT ALL, not merely a numerically hard one.

    Confirmed directly: for a bus with only a CPL attached, the bus's
    own KCL equation is C*dv/dt = -P/v_eff(v), which has a nonzero
    right-hand side for every finite v > 0 (v_eff is always positive by
    the smoothing formula in ConstantPowerLoad) -- the only way to drive
    this to zero is v -> infinity. Newton's method (fsolve), given no
    finite root exists, drives the corresponding state toward float64's
    overflow boundary (~1.8e308, confirmed reproducing a diverged value
    of 1.878e154 -- exactly where squaring it inside the CPL's own v_eff
    smoothing formula overflows) rather than failing cleanly.

    This raises a clear, specific error identifying the disconnected
    island and what it's missing, BEFORE attempting the equilibrium
    solve, rather than letting Newton's method run into a problem that
    has no solution to converge to.
    """
    # Union-find over buses, connected via lines.
    parent = {bus_id: bus_id for bus_id in net.buses}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for line in net.branches:
        union(line.from_bus, line.to_bus)

    islands: dict = {}
    for bus_id in net.buses:
        islands.setdefault(find(bus_id), set()).add(bus_id)

    for island_buses in islands.values():
        has_source = False
        has_load = False
        for comp in net.components:
            if comp.bus in island_buses:
                if isinstance(comp, (Converter, IdealSource)):
                    has_source = True
                elif isinstance(comp, (ConstantPowerLoad, ConstantImpedanceLoad, ConstantCurrentLoad)):
                    has_load = True
        if has_load and not has_source:
            raise ValueError(
                f"network topology is invalid: bus(es) {sorted(island_buses)} form an island with only "
                f"load(s) and no converter or source -- nothing can supply the current those loads draw, "
                f"so no finite equilibrium exists for this island (confirmed: an isolated CPL-only bus drives "
                f"its own voltage state toward infinity rather than converging). Add a converter, an ideal "
                f"source, or a transmission line connecting this island to one that has a source."
            )


def _build_network_from_spec(spec: dict) -> tuple:
    """
    Builds a real `Network` from a user-supplied JSON specification --
    the backend half of the Project Builder. Returns (network,
    input_component_id) -- see the input-selection block below for why
    the id is resolved explicitly here rather than left to
    AutomaticModelBuilder's own (order-dependent, silently ambiguous
    for multi-input networks) default.

    Deliberately restricted to component types that already exist in
    the validated component library (network.components,
    network.converter_topologies, network.controller): three load
    types (CPL, constant impedance, constant current), three converter
    topologies (buck, boost, buck-boost), two controllers (constant
    duty, a minimal PI voltage regulator), a fixed-voltage source, and
    R-L transmission lines.

    Explicitly NOT supported, by honest omission rather than a fabricated
    stand-in: ZIP loads, dynamic/time-varying load profiles, battery/PV/
    wind/fuel-cell source models, bidirectional or generic AC/DC-DC/AC
    converter topologies, and Backstepping/Sliding-Mode/MPC/general-LQR
    controllers -- none of these have a validated physical model in the
    codebase yet, and offering them here would mean either silently
    reusing the wrong physics or fabricating a placeholder that looks
    like a real result but isn't. Requesting one raises a clear error
    naming what's missing, rather than approximating it.

    Once assembled, the resulting system is a genuine DynamicalSystem
    via the same AutomaticModelBuilder every converter_topology and
    network_mrc project already uses -- so equilibrium solving, IMS
    manifold tracing, and recoverability assessment work on it exactly
    as they do on a hand-defined example project, unmodified.
    """
    net = Network(spec.get("name", "custom_network"))

    for b in spec.get("buses", []):
        if b.get("type") == "dynamic":
            if "C" not in b:
                raise ValueError(f"bus '{b.get('id')}': dynamic buses require C (shunt capacitance)")
            net.add_bus(Bus(id=b["id"], C=float(b["C"]), v_init=float(b.get("v_init", 1.0))))
        elif b.get("type") == "ideal":
            if "v_fixed" not in b:
                raise ValueError(f"bus '{b.get('id')}': ideal buses require v_fixed")
            net.add_bus(Bus(id=b["id"], v_fixed=float(b["v_fixed"])))
        else:
            raise ValueError(f"bus '{b.get('id')}': type must be 'dynamic' or 'ideal', got {b.get('type')!r}")

    for ln in spec.get("lines", []):
        net.add_line(Line(id=ln["id"], from_bus=ln["from_bus"], to_bus=ln["to_bus"],
                           R=float(ln["R"]), L=float(ln["L"]), i_init=float(ln.get("i_init", 0.0))))

    for ld in spec.get("loads", []):
        kind = ld.get("type")
        if kind == "cpl":
            net.add_component(ConstantPowerLoad(id=ld["id"], bus=ld["bus"], P=float(ld["P"]),
                                                  v_floor=float(ld.get("v_floor", 0.05))))
        elif kind == "impedance":
            net.add_component(ConstantImpedanceLoad(id=ld["id"], bus=ld["bus"], R=float(ld["R"])))
        elif kind == "current":
            net.add_component(ConstantCurrentLoad(id=ld["id"], bus=ld["bus"], I=float(ld["I"])))
        else:
            raise ValueError(
                f"load '{ld.get('id')}': unsupported type {kind!r} -- supported types are "
                f"{sorted(CUSTOM_LOAD_TYPES)}. ZIP loads and time-varying load profiles are not yet "
                f"implemented in the component library (no validated model exists for them)."
            )

    for src in spec.get("sources", []):
        net.add_component(IdealSource(id=src["id"], bus=src["bus"],
                                       v_source=float(src["v_source"]), R_source=float(src["R_source"])))

    for conv in spec.get("converters", []):
        topology = conv.get("topology")
        if topology not in CUSTOM_CONVERTER_TOPOLOGIES:
            raise ValueError(
                f"converter '{conv.get('id')}': unsupported topology {topology!r} -- supported topologies are "
                f"{sorted(CUSTOM_CONVERTER_TOPOLOGIES)}. Bidirectional, generic DC/AC, and AC/DC converter "
                f"models are not yet implemented in the component library."
            )
        em_cls = CUSTOM_CONVERTER_TOPOLOGIES[topology]
        em_params = conv.get("params", {})
        electrical_model = em_cls(v_in=float(em_params["v_in"]), L=float(em_params["L"]), R_L=float(em_params.get("R_L", 0.0)))

        ctrl_kind = conv.get("controller")
        ctrl_params = conv.get("controller_params", {})
        if ctrl_kind == "constant_duty":
            controller = ConstantDutyController(d=float(ctrl_params.get("d", 0.5)))
        elif ctrl_kind == "pi":
            v_nom_val = float(ctrl_params["v_nom"])
            v_in_val = float(em_params.get("v_in", 0.0))
            default_d_nominal = min(max(v_nom_val / v_in_val, 0.05), 0.95) if abs(v_in_val) > 1e-9 else 0.5
            controller = PIDutyController(
                v_nom=v_nom_val, Kp=float(ctrl_params.get("Kp", 0.3)), Ki=float(ctrl_params.get("Ki", 5.0)),
                d_nominal=float(ctrl_params.get("d_nominal", default_d_nominal)),
            )
        elif ctrl_kind == "droop":
            v_nom_val = float(ctrl_params["v_nom"])
            v_in_val = float(em_params.get("v_in", 0.0))
            default_d_nominal = min(max(v_nom_val / v_in_val, 0.05), 0.95) if abs(v_in_val) > 1e-9 else 0.5
            controller = DroopController(
                v_nom=v_nom_val, Kp=float(ctrl_params.get("Kp", 0.3)), R_droop=float(ctrl_params.get("R_droop", 0.5)),
                d_nominal=float(ctrl_params.get("d_nominal", default_d_nominal)),
            )
        else:
            raise ValueError(
                f"converter '{conv.get('id')}': unsupported controller {ctrl_kind!r} -- supported controllers "
                f"here are {sorted(CUSTOM_CONTROLLER_TYPES)}. Backstepping, sliding-mode, MPC, and a general "
                f"per-converter LQR/MRC assignment are not yet implemented for arbitrary user-built topologies "
                f"(genuine MRC synthesis for a NEW topology requires deriving it with control.mrc_synthesis, "
                f"which is a symbolic-engine step, not something this endpoint can do generically)."
            )
        net.add_component(Converter(id=conv["id"], bus=conv["bus"], electrical_model=electrical_model, controller=controller))

    # Real bug found during validation, fixed here: both Converter and
    # ConstantPowerLoad declare has_input=True (network.converter,
    # network.components). Network.find_input_component() defaults to
    # "the first component with has_input=True" -- for a hand-defined
    # example project there is only ever one such component, so this is
    # unambiguous, but a user-built network can easily have several
    # (multiple converters, multiple CPLs), and silently picking
    # whichever happened to be added first routes the exogenous input
    # to the WRONG component without any error -- confirmed by
    # deliberately reproducing it: a 2-bus boost+CPL network where the
    # CPL was added before the converter had its P silently overridden
    # by the duty-ratio input, producing a self-consistent but wrong
    # equilibrium (KCL still balanced, at the wrong operating point).
    # The fix is to make the choice explicit and required whenever it's
    # ambiguous, rather than order-dependent.
    input_candidates = [c.id for c in net.components if getattr(c, "has_input", False)]
    requested_input_id = spec.get("input_component_id")
    if requested_input_id is not None:
        if requested_input_id not in input_candidates:
            raise ValueError(f"input_component_id '{requested_input_id}' is not a component with an exogenous input "
                              f"(components with an input: {input_candidates})")
    elif len(input_candidates) > 1:
        raise ValueError(f"network has {len(input_candidates)} components that could serve as the exogenous input "
                          f"({input_candidates}) -- 'input_component_id' must be specified explicitly to avoid "
                          f"silently picking one (this is exactly the ambiguity that produced a wrong, but "
                          f"self-consistent, equilibrium during this feature's own validation).")
    elif len(input_candidates) == 0:
        raise ValueError("network has no component with an exogenous input (at least one Converter or "
                          "ConstantPowerLoad is required for the staged pipeline's disturbance/continuation input)")

    resolved_input_id = requested_input_id if requested_input_id is not None else input_candidates[0]
    _check_network_connectivity(net)
    return net, resolved_input_id


TOPOLOGY_REGISTRY = {"buck": BuckModel, "boost": BoostModel, "buckboost": BuckBoostModel}

PROJECTS = {
    "grid_forming_inverter": dict(
        label="Grid-Forming Inverter", category="single_model", model_key="grid_forming_inverter",
        description="Droop-controlled single converter, swing-type dynamics.",
        state_short=["\u03b4", "\u03c9"], input_label="P_set \u2014 active-power setpoint (p.u.)",
        network_summary={"buses": 2, "lines": 1, "converters": 1, "loads": 1, "controllers": 1},
        component_chain=["Grid-Forming Inverter (droop-controlled EMF source)", "Local load P_load", "Line reactance X", "Infinite bus (grid)"],
        controller_description="None by default (open loop, self-stabilising via droop dynamics already in the model); Scheduled LQR available (Conventional Control Library).",
        governing_equations=[
            "d\u03b4/dt = \u03c9",
            "d\u03c9/dt = (1/\u03c4p)\u00b7( -\u03c9 + mp\u00b7(P_set - P_load - P_e(\u03b4)) )",
            "P_e(\u03b4) = (E\u00b7V/X)\u00b7sin(\u03b4)",
        ],
        engineering_outputs=[
            {"name": "Power angle \u03b4", "kind": "state"},
            {"name": "Frequency deviation \u03c9", "kind": "state"},
            {"name": "Electrical power P_e", "kind": "derived (P_e = (E\u00b7V/X)\u00b7sin(\u03b4))"},
        ],
        params=[
            {"key": "E", "label": "EMF magnitude E (p.u.)", "def": 1.0, "help": "Inverter internal EMF magnitude. Typical 0.9\u20131.1"},
            {"key": "V", "label": "Grid voltage V (p.u.)", "def": 1.0, "help": "Grid bus voltage magnitude. Typical 0.9\u20131.1"},
            {"key": "X", "label": "Line reactance X (p.u.)", "def": 0.3, "help": "Reactance between inverter and grid. Typical 0.1\u20130.5"},
            {"key": "tau_p", "label": "Droop time constant \u03c4p (s)", "def": 0.05, "help": "Active-power droop filter time constant."},
            {"key": "m_p", "label": "Droop gain mp", "def": 1.0, "help": "Power/frequency droop gain. Typical 0.5\u20135"},
            {"key": "P_load", "label": "Local load P_load (p.u.)", "def": 0.1, "help": "Active power drawn locally at the inverter's own terminal bus, in addition to power exported through the line. 0 = no local load."},
        ],
        default_input=[0.5], default_state_guess=[0.3, 0.0], sweep_range=[0.0, 0.95], sweep_points=60,
        disturbance_type="offset", default_disturbance=[1.0, 2.0], default_horizon=10.0,
        default_radius=2.0, default_samples=120, default_tol=0.15,
        q_default=[8, 2], r_default=[0.05], u_min_default=[-0.2], u_max_default=[1.2],
    ),
    "dc_microgrid_cpl": dict(
        label="DC Microgrid (Constant-Power Load)", category="single_model", model_key="dc_microgrid_cpl",
        description="Bistable large-signal case (classic CPL instability).",
        state_short=["i", "v"], input_label="P_load \u2014 constant-power load (p.u.)",
        network_summary={"buses": 2, "lines": 1, "converters": 1, "loads": 1, "controllers": 0},
        component_chain=["Ideal DC source (Vin)", "R-L line", "DC bus (capacitance C)", "Constant-power load"],
        controller_description="None by default (open loop); Scheduled LQR available (Conventional Control Library).",
        params=[
            {"key": "L", "label": "Inductance L (H)", "def": 0.005, "help": "Line/converter inductance."},
            {"key": "C", "label": "Capacitance C (F)", "def": 0.05, "help": "DC bus capacitance."},
            {"key": "r", "label": "Resistance / damping r (p.u.)", "def": 0.15, "help": "Middlebrook stability criterion applies."},
            {"key": "Vin", "label": "Source voltage Vin (p.u.)", "def": 1.2, "help": "Source-side regulated voltage."},
            {"key": "v_floor", "label": "Voltage floor (p.u.)", "def": 0.05, "help": "Numerical smoothing floor. Leave at default."},
        ],
        default_input=[0.5], default_state_guess=[0.44, 1.13], sweep_range=[0.05, 1.2], sweep_points=60,
        disturbance_type="absolute", default_disturbance=[0.68894065, 0.29546767], default_horizon=3.0,
        default_radius=0.9, default_samples=120, default_tol=0.1,
        q_default=[1, 25], r_default=[0.5], u_min_default=[0.05], u_max_default=[1.5],
    ),
    "buck_converter": dict(
        label="Buck Converter", category="converter_topology", topology="buck",
        description="Averaged CCM buck converter feeding a resistive load. Validated against the textbook ratio V_o/V_in = D.",
        state_short=["v_out", "i_L"], input_label="D \u2014 switch duty ratio",
        params=[
            {"key": "v_in", "label": "Input voltage V_in (V)", "def": 12.0, "help": "Idealised, stiff input rail."},
            {"key": "L", "label": "Inductance L (H)", "def": 1e-3, "help": "Converter inductor."},
            {"key": "R_L", "label": "Inductor ESR R_L (\u03a9)", "def": 0.05, "help": "Small nonzero value avoids a near-singular Jacobian."},
            {"key": "R_load", "label": "Load resistance (\u03a9)", "def": 5.0, "help": "Constant-impedance load."},
            {"key": "C", "label": "Output capacitance C (F)", "def": 0.02, "help": "Output bus capacitance."},
        ],
        default_input=[0.4], default_state_guess=None, sweep_range=[0.1, 0.85], sweep_points=150,
        disturbance_type="offset", default_disturbance=[0.5, -0.2], default_horizon=0.15,
        default_radius=1.0, default_samples=60, default_tol=0.1,
    ),
    "boost_converter": dict(
        label="Boost Converter", category="converter_topology", topology="boost",
        description="Averaged CCM boost converter feeding a resistive load. Validated against the textbook ratio V_o/V_in = 1/(1-D).",
        state_short=["v_out", "i_L"], input_label="D \u2014 switch duty ratio",
        params=[
            {"key": "v_in", "label": "Input voltage V_in (V)", "def": 12.0, "help": "Idealised, stiff input rail."},
            {"key": "L", "label": "Inductance L (H)", "def": 1e-3, "help": "Converter inductor."},
            {"key": "R_L", "label": "Inductor ESR R_L (\u03a9)", "def": 0.05, "help": "Small nonzero value avoids a near-singular Jacobian."},
            {"key": "R_load", "label": "Load resistance (\u03a9)", "def": 20.0, "help": "Constant-impedance load."},
            {"key": "C", "label": "Output capacitance C (F)", "def": 0.02, "help": "Output bus capacitance."},
        ],
        default_input=[0.5], default_state_guess=None, sweep_range=[0.1, 0.8], sweep_points=150,
        disturbance_type="offset", default_disturbance=[1.0, -0.3], default_horizon=0.15,
        default_radius=1.5, default_samples=60, default_tol=0.1,
    ),
    "buckboost_converter": dict(
        label="Buck-Boost Converter", category="converter_topology", topology="buckboost",
        description="Averaged CCM (inverting, output-magnitude convention) buck-boost converter. Validated against V_o/V_in = D/(1-D).",
        state_short=["v_out", "i_L"], input_label="D \u2014 switch duty ratio",
        params=[
            {"key": "v_in", "label": "Input voltage V_in (V)", "def": 12.0, "help": "Idealised, stiff input rail."},
            {"key": "L", "label": "Inductance L (H)", "def": 1e-3, "help": "Converter inductor."},
            {"key": "R_L", "label": "Inductor ESR R_L (\u03a9)", "def": 0.05, "help": "Small nonzero value avoids a near-singular Jacobian."},
            {"key": "R_load", "label": "Load resistance (\u03a9)", "def": 8.0, "help": "Constant-impedance load."},
            {"key": "C", "label": "Output capacitance C (F)", "def": 0.02, "help": "Output bus capacitance."},
        ],
        default_input=[0.4], default_state_guess=None, sweep_range=[0.1, 0.75], sweep_points=150,
        disturbance_type="offset", default_disturbance=[0.6, -0.2], default_horizon=0.15,
        default_radius=1.0, default_samples=60, default_tol=0.1,
    ),
    "network_auto_mrc": dict(
        label="Network + Auto-MRC Demo", category="network_mrc",
        description="A network assembled from primitives, driven by a Manifold-Reshaping Control law derived automatically by the Symbolic Engine -- no linearisation, no hand-tuned gain schedule.",
        state_short=["v_bus", "i_L", "v_o"],
        params=[
            {"key": "R", "label": "Line resistance R (\u03a9)", "def": 0.20, "help": ""},
            {"key": "L", "label": "Line inductance L (H)", "def": 1.5e-3, "help": ""},
            {"key": "C", "label": "Bus capacitance C (F)", "def": 2.5e-3, "help": ""},
            {"key": "P", "label": "CPL power P (W)", "def": 10e3, "help": ""},
        ],
        default_km=500.0, default_v_bus_init=400.0, default_disturbance=[-5.0, 0.0, 0.0], default_horizon=0.02,
    ),
}


def _network_summary_from_network(network: Network) -> dict:
    """
    Computed directly from the real, assembled Network object (not a
    hand-maintained description) -- for the two categories that build a
    genuine Network (converter_topology, network_mrc), this is a live
    property of the model, not documentation that could drift out of sync
    with it.

    Component counts are grouped by each component's own Python class
    name (e.g. "ConstantPowerLoad", "Converter"), discovered dynamically
    rather than assumed from a fixed list. Converters are additionally
    broken down by their ElectricalModel's class name (BuckModel,
    BoostModel, ConverterCPLElectricalModel, ...) and control_mode
    attribute (grid_forming, grid_following, passive) -- both already
    declared on every ElectricalModel (network.electrical_model), so
    this is real data, not inference. Today's library only implements
    grid_forming control_mode; the breakdown will show that plainly
    (all converters counted as grid-forming) rather than fabricate a
    grid-following count that doesn't exist. A future component type (a
    battery, a PV source, a synchronous machine) appears here
    automatically the moment it exists in the component library and is
    used in a network, with no changes needed to this function.
    """
    n_converters = sum(1 for c in network.components if isinstance(c, Converter))
    type_counts: dict = {}
    topology_counts: dict = {}
    control_mode_counts: dict = {}
    for c in network.components:
        type_counts[type(c).__name__] = type_counts.get(type(c).__name__, 0) + 1
        if isinstance(c, Converter):
            em_name = type(c.electrical_model).__name__
            topology_counts[em_name] = topology_counts.get(em_name, 0) + 1
            mode = getattr(c.electrical_model, "control_mode", "unknown")
            control_mode_counts[mode] = control_mode_counts.get(mode, 0) + 1
    n_loads = sum(1 for c in network.components if isinstance(c, (ConstantPowerLoad, ConstantImpedanceLoad)))
    return {
        "buses": len(network.buses), "lines": len(network.branches),
        "converters": n_converters, "loads": n_loads, "controllers": n_converters,
        "component_types": type_counts,
        "converter_topologies": topology_counts,
        "converter_control_modes": control_mode_counts,
    }


def _topology_tree_from_network(network: Network) -> list:
    """
    Bus-by-bus attachment tree, e.g.:
        Bus 'bus' (dynamic, C=0.0025)
          |-- Converter 'conv'
          |-- ConstantPowerLoad 'cpl'
    Built directly from the real Network object's bus/component
    associations (network.components_at), so it reflects whatever
    topology was actually assembled, not a fixed template.
    """
    lines = []
    for bus in network.buses.values():
        kind = f"dynamic, C={bus.C}" if bus.is_dynamic else f"ideal, v_fixed={bus.v_fixed}"
        lines.append(f"Bus '{bus.id}' ({kind})")
        attached = network.components_at(bus.id)
        for i, comp in enumerate(attached):
            branch_char = "\\--" if i == len(attached) - 1 else "|--"
            lines.append(f"  {branch_char} {type(comp).__name__} '{comp.id}'")
        if not attached:
            lines.append("  (no components attached directly; connected via line(s) only)")
    return lines


def _component_chain_from_network(network: Network) -> list:
    chain = []
    for bus in network.buses.values():
        if not bus.is_dynamic:
            chain.append(f"Ideal source bus '{bus.id}' (v_fixed={bus.v_fixed})")
    for comp in network.components:
        if isinstance(comp, Converter):
            chain.append(f"Converter '{comp.id}' ({type(comp.electrical_model).__name__}, {type(comp.controller).__name__})")
        else:
            chain.append(f"{type(comp).__name__} '{comp.id}' (bus: {comp.bus})")
    for bus in network.buses.values():
        if bus.is_dynamic:
            chain.append(f"Dynamic bus '{bus.id}' (C={bus.C})")
    return chain


def _controller_description(project: dict, network=None) -> str:
    if project["category"] == "network_mrc":
        return "IMS-Native MRC (Symbolic Engine derived, automatically computed -- see Build Model stage for the exact law)."
    if project["category"] == "converter_topology":
        return "Constant Duty Cycle (open-loop reference -- D is fixed, not actively regulated by feedback)."
    if project["category"] == "custom_network" and network is not None:
        parts = []
        for c in network.components:
            if isinstance(c, Converter):
                parts.append(f"{c.id}: {type(c.controller).__name__}")
        return "; ".join(parts) if parts else "No converters with an assigned controller in this network."
    return project.get("controller_description", "None by default (open loop).")


def _sanitize_for_json(obj):
    """
    Recursively replaces float('nan'), float('inf'), and float('-inf')
    with None (JSON null) throughout any nested dict/list/tuple
    structure. Needed because Python's default json encoder (which
    Flask's jsonify uses) happily emits the literal tokens `NaN`,
    `Infinity`, and `-Infinity` -- valid Python float reprs, but NOT
    valid JSON per RFC 8259 -- which a strict browser JSON.parse()
    correctly rejects rather than silently accepting.

    Confirmed reproducible (not hypothetical): an extreme-but-plausible
    user-entered converter inductance (e.g. L=1e-80) converges to a
    genuinely valid equilibrium (residual ~2.8e-14, easily passing the
    existing convergence check) while jacobian_condition_number
    overflows to literal inf during the post-hoc SVD-based condition
    number computation -- a case the residual-based convergence check
    alone cannot catch, since the equilibrium point itself is fine and
    only an auxiliary diagnostic overflows.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


class _SanitizingJSONProvider(DefaultJSONProvider):
    """Applies _sanitize_for_json to every response body before serialization, application-wide."""

    def dumps(self, obj, **kwargs):
        return super().dumps(_sanitize_for_json(obj), **kwargs)


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)
    app.json = _SanitizingJSONProvider(app)
    register_auth_routes(app)

    @app.route("/")
    def index():
        return send_from_directory(STATIC_DIR, "explorer.html")

    @app.route("/assets/<path:filename>")
    def assets(filename):
        return send_from_directory(os.path.join(STATIC_DIR, "assets"), filename)

    @app.route("/api/health")
    def health():
        import ims_platform
        return jsonify(status="ok", version=ims_platform.__version__)

    @app.route("/api/projects")
    def list_projects():
        return jsonify([
            {"id": pid, "label": p["label"], "category": p["category"], "description": p["description"]}
            for pid, p in PROJECTS.items()
        ])

    @app.route("/api/project/<project_id>")
    def project_detail(project_id):
        if project_id not in PROJECTS:
            return jsonify(error=f"unknown project '{project_id}'"), 404
        return jsonify(dict(PROJECTS[project_id], id=project_id))

    def _wrap(single_fn, network_mrc_fn):
        def handler(project_id):
            if project_id not in PROJECTS:
                return jsonify(error=f"unknown project '{project_id}'"), 404
            payload = request.get_json(force=True) or {}
            try:
                if PROJECTS[project_id]["category"] == "network_mrc":
                    return jsonify(network_mrc_fn(payload))
                return jsonify(single_fn(PROJECTS[project_id], payload))
            except Exception as e:
                return jsonify(error=str(e)), 400
        return handler

    app.add_url_rule("/api/project/<project_id>/build", "build",
                      _wrap(stage_build, network_mrc_build), methods=["POST"])
    app.add_url_rule("/api/project/<project_id>/equilibrium", "equilibrium",
                      _wrap(stage_equilibrium, network_mrc_equilibrium), methods=["POST"])
    app.add_url_rule("/api/project/<project_id>/simulate", "simulate",
                      _wrap(stage_simulate, network_mrc_simulate), methods=["POST"])
    app.add_url_rule("/api/project/<project_id>/ims_analysis", "ims_analysis",
                      _wrap(stage_ims_analysis, network_mrc_ims_analysis), methods=["POST"])

    # Network + Auto-MRC: attach the synthesized MRC to the converter in the
    # assembled network, then rerun baseline vs MRC on that identical network.
    def _network_mrc_route(fn):
        payload = request.get_json(force=True) or {}
        try:
            return jsonify(fn(payload))
        except Exception as e:
            return jsonify(error=str(e)), 400

    app.add_url_rule("/api/network_mrc/apply", "network_mrc_apply_route",
                      lambda: _network_mrc_route(network_mrc_apply), methods=["POST"])
    app.add_url_rule("/api/network_mrc/closed_loop", "network_mrc_closed_loop_route",
                      lambda: _network_mrc_route(network_mrc_closed_loop), methods=["POST"])
    app.add_url_rule("/api/network_mrc/inspect", "network_mrc_inspect_route",
                      lambda: _network_mrc_route(network_mrc_inspect), methods=["POST"])

    # General Network-Builder MRC capability: runs the real automatic
    # derivation pipeline (mrc_designer.auto_manifold) on an ARBITRARY
    # user-built network_spec, rather than the fixed Converter-CPL demo
    # network the /apply and /closed_loop routes above are scoped to.
    app.add_url_rule("/api/network_mrc/auto_apply", "network_mrc_auto_apply_route",
                      lambda: _network_mrc_route(network_mrc_auto_apply), methods=["POST"])
    app.add_url_rule("/api/network_mrc/auto_closed_loop", "network_mrc_auto_closed_loop_route",
                      lambda: _network_mrc_route(network_mrc_auto_closed_loop), methods=["POST"])

    def _custom_handler(stage_fn):
        def handler(stage_ignored=None):
            def _smart_default_input(network_spec):
                # A PI controller's control_signal uses "p['v_nom'] if u
                # is None else u" -- u OVERRIDES the fixed v_nom whenever
                # provided (this override exists so a manifold sweep can
                # vary u deliberately). But for a plain equilibrium solve
                # where the user hasn't touched the input field, silently
                # defaulting to a generic duty-scale constant (0.5) would
                # override a PI converter's real voltage setpoint (e.g.
                # 24) with 0.5 -- producing a nonsensical, wildly
                # saturated duty command. Default to that converter's own
                # v_nom instead, so the "override" is a no-op unless the
                # user deliberately changes it.
                for conv in (network_spec or {}).get("converters", []):
                    if conv.get("controller") in ("pi", "droop"):
                        v_nom = conv.get("controller_params", {}).get("v_nom")
                        if v_nom is not None:
                            return [float(v_nom)]
                return [0.5]

            payload = request.get_json(force=True) or {}
            spec = payload.get("network_spec")
            if not spec:
                return jsonify(error="request body must include 'network_spec'"), 400
            synthetic_project = {
                "label": spec.get("name", "Custom Network"), "category": "custom_network",
                "description": "User-built network (Project Builder).",
                "default_input": payload.get("nominal_input", _smart_default_input(spec)),
                "default_horizon": payload.get("horizon", 1.0),
                "default_radius": payload.get("radius", 1.0),
                "default_samples": payload.get("n_samples", 60),
                "default_tol": payload.get("recovery_tol", 0.1),
                "sweep_range": payload.get("sweep_range", [0.0, 1.0]),
                "sweep_points": payload.get("sweep_points", 60),
            }
            try:
                return jsonify(stage_fn(synthetic_project, payload))
            except Exception as e:
                return jsonify(error=str(e)), 400
        return handler

    app.add_url_rule("/api/custom_project/build", "custom_build",
                      _custom_handler(stage_build), methods=["POST"])
    app.add_url_rule("/api/custom_project/equilibrium", "custom_equilibrium",
                      _custom_handler(stage_equilibrium), methods=["POST"])
    app.add_url_rule("/api/custom_project/simulate", "custom_simulate",
                      _custom_handler(stage_simulate), methods=["POST"])
    app.add_url_rule("/api/custom_project/ims_analysis", "custom_ims_analysis",
                      _custom_handler(stage_ims_analysis), methods=["POST"])

    # ------------------------------------------------------------------
    # Iberian 2025-Inspired Overvoltage Cascade: the platform's flagship
    # "Case Library" scenario (funding-demo brief). Deliberately its own
    # small set of routes rather than forced through the generic
    # single_model/converter_topology/custom_network staged pipeline: the
    # scenario has a fixed topology (only the four sliders vary), a
    # scripted multi-event disturbance timeline, and a bespoke dashboard
    # (voltage traces, IMS margin, recoverability verdict, boundary map)
    # that doesn't fit the generic build/equilibrium/simulate/ims_analysis
    # shape.
    # ------------------------------------------------------------------
    @app.route("/api/iberian/summary", methods=["GET", "POST"])
    def iberian_summary():
        payload = request.get_json(force=True, silent=True) or {}
        try:
            return jsonify(iberian_scenario.build_summary(payload.get("params", {})))
        except Exception as e:
            return jsonify(error=str(e)), 400

    @app.route("/api/iberian/stress_test", methods=["POST"])
    def iberian_stress_test():
        payload = request.get_json(force=True) or {}
        try:
            result = iberian_scenario.run_stress_test(
                payload.get("params", {}),
                severity=float(payload.get("severity", 1.0)),
                tail_horizon=float(payload.get("tail_horizon", 6.0)),
            )
            return jsonify(result)
        except Exception as e:
            return jsonify(error=str(e)), 400

    @app.route("/api/iberian/boundary_map", methods=["POST"])
    def iberian_boundary_map():
        payload = request.get_json(force=True) or {}
        try:
            result = iberian_scenario.run_boundary_map(
                payload.get("params", {}),
                severities=payload.get("severities"),
                tail_horizon=float(payload.get("tail_horizon", 6.0)),
            )
            return jsonify(result)
        except Exception as e:
            return jsonify(error=str(e)), 400

    @app.route("/api/iberian/boundary_map_2d", methods=["POST"])
    def iberian_boundary_map_2d():
        payload = request.get_json(force=True) or {}
        try:
            result = iberian_scenario.run_boundary_map_2d(
                payload.get("params", {}),
                x_param=payload.get("x_param", "KQ"),
                x_values=payload.get("x_values"),
                severities=payload.get("severities"),
                tail_horizon=float(payload.get("tail_horizon", 6.0)),
            )
            return jsonify(result)
        except Exception as e:
            return jsonify(error=str(e)), 400

    @app.route("/api/iberian/gfm_comparison", methods=["POST"])
    def iberian_gfm_comparison():
        payload = request.get_json(force=True) or {}
        try:
            result = iberian_scenario.run_gfm_comparison(
                payload.get("params", {}),
                gfm_fractions=payload.get("gfm_fractions"),
                severities=payload.get("severities"),
                tail_horizon=float(payload.get("tail_horizon", 8.0)),
            )
            return jsonify(result)
        except Exception as e:
            return jsonify(error=str(e)), 400

    @app.route("/api/iberian/pre_incident_warning", methods=["POST"])
    def iberian_pre_incident_warning():
        payload = request.get_json(force=True) or {}
        try:
            stress_result = iberian_scenario.run_stress_test(
                payload.get("params", {}),
                severity=float(payload.get("severity", 1.0)),
                tail_horizon=float(payload.get("tail_horizon", 8.0)),
            )
            warning = iberian_scenario.analyze_pre_incident_warning(
                stress_result, warning_margin=float(payload.get("warning_margin", 0.03)),
            )
            ims_warning = None
            if payload.get("include_ims_warning"):
                ims_geo = iberian_scenario.run_ims_geometry_analysis(
                    payload.get("params", {}), severity=float(payload.get("severity", 1.0)),
                    roa_n_v=3, roa_n_timer=3,
                )
                ims_warning = iberian_scenario.analyze_pre_incident_warning_ims_geometry(
                    stress_result, ims_geo, d_M_threshold=float(payload.get("d_M_threshold", 0.05)))
            return jsonify({"stress_test": stress_result, "warning": warning, "ims_warning": ims_warning})
        except Exception as e:
            return jsonify(error=str(e)), 400

    @app.route("/api/iberian/recommendations", methods=["POST"])
    def iberian_recommendations():
        payload = request.get_json(force=True) or {}
        try:
            result = iberian_scenario.find_recommended_interventions(
                payload.get("params", {}),
                severity=float(payload.get("severity", 1.0)),
                tail_horizon=float(payload.get("tail_horizon", 8.0)),
            )
            return jsonify(result)
        except Exception as e:
            return jsonify(error=str(e)), 400

    @app.route("/api/iberian/ims_geometry", methods=["POST"])
    def iberian_ims_geometry():
        payload = request.get_json(force=True, silent=True) or {}
        try:
            result = iberian_scenario.run_ims_geometry_analysis(
                payload.get("params", {}),
                severity=float(payload.get("severity", 1.0)),
                roa_n_v=int(payload.get("roa_n_v", 8)),
                roa_n_timer=int(payload.get("roa_n_timer", 6)),
            )
            return jsonify(result)
        except Exception as e:
            return jsonify(error=str(e)), 400

    @app.route("/api/iberian/counterfactual", methods=["POST"])
    def iberian_counterfactual():
        payload = request.get_json(force=True, silent=True) or {}
        try:
            result = iberian_scenario.run_baseline_vs_dynamic_q_counterfactual(
                severity=float(payload.get("severity", 1.0)),
                tail_horizon=float(payload.get("tail_horizon", 6.0)),
            )
            return jsonify(result)
        except Exception as e:
            return jsonify(error=str(e)), 400

    @app.route("/api/case_library", methods=["GET"])
    def case_library():
        return jsonify(iberian_scenario.CASE_LIBRARY)

    # ------------------------------------------------------------------
    # GFM Current-Limit Recovery
    # ------------------------------------------------------------------
    @app.route("/api/gfm_current_limit/stress_test", methods=["POST"])
    def gfm_current_limit_stress_test():
        payload = request.get_json(force=True, silent=True) or {}
        try:
            return jsonify(gfm_current_limit_case.run_stress_test(payload.get("params", {})))
        except Exception as e:
            return jsonify(error=str(e)), 400

    @app.route("/api/gfm_current_limit/sweep", methods=["POST"])
    def gfm_current_limit_sweep():
        payload = request.get_json(force=True, silent=True) or {}
        try:
            return jsonify(gfm_current_limit_case.run_imax_sweep(payload.get("values")))
        except Exception as e:
            return jsonify(error=str(e)), 400

    @app.route("/api/gfm_current_limit/ims_geometry", methods=["POST"])
    def gfm_current_limit_ims_geometry():
        payload = request.get_json(force=True, silent=True) or {}
        try:
            return jsonify(gfm_current_limit_case.run_ims_geometry_analysis(payload.get("params", {})))
        except Exception as e:
            return jsonify(error=str(e)), 400

    @app.route("/api/gfm_current_limit/recommendations", methods=["POST"])
    def gfm_current_limit_recommendations():
        payload = request.get_json(force=True, silent=True) or {}
        try:
            return jsonify(gfm_current_limit_case.find_recommended_interventions(payload.get("params", {})))
        except Exception as e:
            return jsonify(error=str(e)), 400

    # ------------------------------------------------------------------
    # Multi-Converter Fault Recovery
    # ------------------------------------------------------------------
    @app.route("/api/multi_converter_fault/stress_test", methods=["POST"])
    def multi_converter_fault_stress_test():
        payload = request.get_json(force=True, silent=True) or {}
        try:
            return jsonify(multi_converter_fault_case.run_stress_test(payload.get("params", {})))
        except Exception as e:
            return jsonify(error=str(e)), 400

    @app.route("/api/multi_converter_fault/asymmetry", methods=["POST"])
    def multi_converter_fault_asymmetry():
        try:
            return jsonify(multi_converter_fault_case.verify_local_vs_remote_support_asymmetry())
        except Exception as e:
            return jsonify(error=str(e)), 400

    @app.route("/api/multi_converter_fault/ims_geometry", methods=["POST"])
    def multi_converter_fault_ims_geometry():
        payload = request.get_json(force=True, silent=True) or {}
        try:
            return jsonify(multi_converter_fault_case.run_ims_geometry_analysis(payload.get("params", {})))
        except Exception as e:
            return jsonify(error=str(e)), 400

    @app.route("/api/multi_converter_fault/recommendations", methods=["POST"])
    def multi_converter_fault_recommendations():
        payload = request.get_json(force=True, silent=True) or {}
        try:
            return jsonify(multi_converter_fault_case.find_recommended_interventions(payload.get("params", {})))
        except Exception as e:
            return jsonify(error=str(e)), 400

    # ------------------------------------------------------------------
    # Final Report generation (single engine, HTML and PDF from the
    # same markup) -- see server/report_generator.py's module docstring.
    # ------------------------------------------------------------------
    def _build_report_html(case_id: str, payload: dict) -> str:
        params = payload.get("params", {})
        if case_id == "iberian_2025_overvoltage_cascade":
            severity = float(payload.get("severity", 1.0))
            result = iberian_scenario.run_stress_test(params, severity=severity)
            cf = iberian_scenario.run_baseline_vs_dynamic_q_counterfactual(severity=severity) \
                if payload.get("include_counterfactual") else None
            bm = iberian_scenario.run_boundary_map(params) if payload.get("include_boundary") else None
            rec = iberian_scenario.find_recommended_interventions(params, severity=severity) \
                if payload.get("include_recommendations") else None
            ims_geo = iberian_scenario.run_ims_geometry_analysis(params, severity=severity, roa_n_v=7, roa_n_timer=5) \
                if payload.get("include_ims_geometry") else None
            ims_warn = iberian_scenario.analyze_pre_incident_warning_ims_geometry(result, ims_geo) \
                if (ims_geo and ims_geo.get("available")) else None
            return report_generator.build_iberian_report(result, counterfactual=cf, boundary=bm, recommendations=rec,
                                                           ims_geometry=ims_geo, ims_warning=ims_warn)
        elif case_id == "gfm_current_limit_recovery":
            result = gfm_current_limit_case.run_stress_test(params)
            sweep = gfm_current_limit_case.run_imax_sweep() if payload.get("include_sweep") else None
            ims_geo = gfm_current_limit_case.run_ims_geometry_analysis(params) if payload.get("include_ims_geometry") else None
            rec = gfm_current_limit_case.find_recommended_interventions(params) if payload.get("include_recommendations") else None
            return report_generator.build_gfm_report(result, sweep=sweep, ims_geometry=ims_geo, recommendations=rec)
        elif case_id == "multi_converter_fault_recovery":
            result = multi_converter_fault_case.run_stress_test(params)
            asym = multi_converter_fault_case.verify_local_vs_remote_support_asymmetry() \
                if payload.get("include_asymmetry") else None
            ims_geo = multi_converter_fault_case.run_ims_geometry_analysis(params) if payload.get("include_ims_geometry") else None
            rec = multi_converter_fault_case.find_recommended_interventions(params) if payload.get("include_recommendations") else None
            return report_generator.build_multiconverter_report(result, asymmetry=asym, ims_geometry=ims_geo, recommendations=rec)
        else:
            raise ValueError(f"Unknown case_id: {case_id!r}")

    @app.route("/api/report/<case_id>/html", methods=["POST"])
    def report_html(case_id):
        payload = request.get_json(force=True, silent=True) or {}
        try:
            html_str = _build_report_html(case_id, payload)
            return Response(html_str, mimetype="text/html")
        except Exception as e:
            return jsonify(error=str(e)), 400

    @app.route("/api/report/<case_id>/pdf", methods=["POST"])
    def report_pdf(case_id):
        payload = request.get_json(force=True, silent=True) or {}
        try:
            html_str = _build_report_html(case_id, payload)
            pdf_bytes = report_generator.html_to_pdf_bytes(html_str)
            return Response(pdf_bytes, mimetype="application/pdf", headers={
                "Content-Disposition": f'inline; filename="{case_id}_report.pdf"'
            })
        except Exception as e:
            return jsonify(error=str(e)), 400

    return app


# ----------------------------------------------------------------------
# Shared construction: builds the real DynamicalSystem for single_model
# and converter_topology projects, so every stage below operates on the
# same object the rest of the platform's test suite already validates.
# ----------------------------------------------------------------------

def _build_system(project: dict, payload: dict):
    """Returns (system, param_name_for_continuation, extra_metadata)."""
    category = project["category"]

    if category == "single_model":
        cls = MODEL_REGISTRY[project["model_key"]]
        params = payload.get("params", {})
        system = cls(params=params)
        return system, f"{project['model_key']}_param", {}

    if category == "custom_network":
        spec = payload.get("network_spec")
        if not spec:
            raise ValueError("custom_network category requires 'network_spec' in the payload")
        net, input_component_id = _build_network_from_spec(spec)
        system = AutomaticModelBuilder.build(net, input_component_id=input_component_id, name=spec.get("name"))
        return system, "custom_param", {"input_component_id": input_component_id}

    if category == "converter_topology":
        params = payload.get("params", {})

        def gv(key):
            return float(params.get(key, next(p["def"] for p in project["params"] if p["key"] == key)))

        v_in, L, R_L, R_load, C = gv("v_in"), gv("L"), gv("R_L"), gv("R_load"), gv("C")
        duty = float(payload.get("nominal_input", project["default_input"])[0])

        net = Network(project["label"])
        v_guess = {"buck": duty * v_in, "boost": v_in / max(1e-6, 1 - duty),
                   "buckboost": duty * v_in / max(1e-6, 1 - duty)}[project["topology"]]
        net.add_bus(Bus(id="out", C=C, v_init=v_guess, v_min=0.05 * max(1.0, v_guess)))
        em_cls = TOPOLOGY_REGISTRY[project["topology"]]
        net.add_component(Converter(id="conv", bus="out", electrical_model=em_cls(v_in=v_in, L=L, R_L=R_L),
                                     controller=ConstantDutyController(d=duty)))
        net.add_component(ConstantImpedanceLoad(id="load", bus="out", R=R_load))
        system = AutomaticModelBuilder.build(net)
        return system, "conv_ctrl_d", {"v_guess": v_guess}

    raise ValueError(f"_build_system does not handle category '{category}'")


def _performance_metrics(t: np.ndarray, x: np.ndarray, x_star: np.ndarray, settling_band: float = 0.05) -> dict:
    """
    Standard time-domain response metrics, computed per state, from a
    trajectory x (shape [n_states, n_time]) relative to its equilibrium
    x_star. Definitions:

        max_deviation[i]      = max_t |x_i(t) - x_star_i|
        steady_state_error[i] = |x_i(T) - x_star_i|  (deviation at the
                                 final simulated time, not a true t->inf
                                 limit -- see Simulation stage note)
        settling_time[i]      = earliest time after which |x_i(t) -
                                 x_star_i| stays within `settling_band`
                                 (default 5%) of max_deviation[i] for the
                                 remainder of the simulated horizon; None
                                 if it never does.
        overshoot_pct[i]      = the largest excursion *past* x_star_i in
                                 the direction opposite the initial
                                 displacement, as a percentage of that
                                 initial displacement. 0 if the response
                                 never crosses x_star_i, or if the state
                                 starts exactly at its equilibrium value.
                                 (This is the standard step-response
                                 overshoot definition, applied to an
                                 initial-condition perturbation rather
                                 than a reference step -- the two
                                 coincide when the "step" is the jump
                                 from the disturbed state to x_star.)
    """
    n_states = x.shape[0]
    dev = x - x_star[:, None]
    abs_dev = np.abs(dev)

    max_deviation = abs_dev.max(axis=1)
    steady_state_error = abs_dev[:, -1]

    settling_time = []
    for i in range(n_states):
        band = settling_band * max_deviation[i] if max_deviation[i] > 0 else 0.0
        outside = np.where(abs_dev[i] > band)[0]
        if len(outside) == 0:
            settling_time.append(0.0)
        elif outside[-1] == len(t) - 1:
            settling_time.append(None)  # never settles within the simulated horizon
        else:
            settling_time.append(float(t[outside[-1] + 1]))

    overshoot_pct = []
    for i in range(n_states):
        initial_dev = dev[i, 0]
        if abs(initial_dev) < 1e-12:
            overshoot_pct.append(0.0)
            continue
        opposite_excursion = -np.sign(initial_dev) * dev[i]
        worst = float(np.max(opposite_excursion))
        overshoot_pct.append(max(0.0, 100.0 * worst / abs(initial_dev)))

    return {
        "max_deviation": max_deviation.tolist(),
        "steady_state_error": steady_state_error.tolist(),
        "settling_time": settling_time,
        "overshoot_pct": overshoot_pct,
    }


def _menger_curvature(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
    """
    Discrete curvature of a curve at point B, estimated from three
    consecutive samples A, B, C via the Menger curvature (the reciprocal
    of the circumradius of triangle ABC, computed from side lengths via
    Heron's formula, so it works in any state-space dimension without
    needing a 2D or 3D embedding). Exact for points sampled from a true
    circular arc (curvature = 1/R), zero for collinear points -- both
    confirmed to machine precision in this module's own validation
    (see tests/test_explorer_ims_depth.py for the same check).
    """
    a = float(np.linalg.norm(B - C))
    b = float(np.linalg.norm(C - A))
    c = float(np.linalg.norm(A - B))
    s = (a + b + c) / 2
    area_sq = s * (s - a) * (s - b) * (s - c)
    if area_sq <= 0 or a * b * c < 1e-15:
        return 0.0
    area = area_sq ** 0.5
    return 4 * area / (a * b * c)


def _manifold_statistics(M, param_name: str, x_star: np.ndarray = None) -> dict:
    """
    Statistics about the traced intrinsic manifold, computed entirely
    from data the continuation already produces (architecture document
    Part II Section 2.3), plus discrete curvature (see
    `_menger_curvature`) and arc length -- no unvalidated numerical
    machinery:

        dimension: always 1 for this platform's current single-parameter
            continuation (architecture document Section 5.4 notes
            multi-parameter continuation, and hence higher-dimensional
            manifolds, as future work) -- stated explicitly rather than
            implied, since a manifold traced by a scalar sweep is a
            curve, not a general submanifold.
        arc_length: sum of consecutive point-to-point distances -- a
            genuine geometric measure of the manifold's "geometric
            complexity" the sweep traced, validated to 3e-6 relative
            error against the exact analytic arc length of a synthetic
            quarter-circle (see tests/test_explorer_manifold_geometry.py).
        tangent_at_nearest: the unit tangent direction to the manifold
            at the traced point closest to x_star, estimated by a
            central (or one-sided, at an endpoint) finite difference
            between neighbouring points -- the direction the operating
            point would move along the manifold under a small change in
            the continuation parameter.
        operating_region_stable: the [min, max] range of the FIRST state
            coordinate across only the stable-branch manifold points --
            a concrete, data-derived candidate for "recommended operating
            region" in that coordinate. Deliberately restricted to what
            the trace itself supports: this is the extent of the stable
            branch actually swept, not a claim about safety margins,
            thermal limits, or any other engineering constraint the
            platform does not model.
        n_stable / n_unstable: direct counts from each point's own
            equilibrium classification (already computed per point).
        singular_points: points where consecutive manifold samples
            disagree on stability -- a discrete detector for where
            normal hyperbolicity is lost along the traced curve
            (architecture document Section 1.6, a fold/termination of
            the fast equilibrium manifold under the continuation
            parameter). This is exact for the sampled points themselves
            and approximate for the true crossing location (reported as
            the midpoint of the two straddling parameter values).
    """
    points = M.points
    n_stable = sum(1 for p in points if p.is_stable)
    n_unstable = len(points) - n_stable

    singular_points = []
    for i in range(1, len(points)):
        if points[i].is_stable != points[i - 1].is_stable:
            singular_points.append({
                "alpha_before": float(points[i - 1].param_value),
                "alpha_after": float(points[i].param_value),
                "alpha_midpoint": float((points[i - 1].param_value + points[i].param_value) / 2),
            })

    curvatures = []
    for i in range(1, len(points) - 1):
        curvatures.append(_menger_curvature(points[i - 1].x_star, points[i].x_star, points[i + 1].x_star))

    arc_length = sum(float(np.linalg.norm(points[i + 1].x_star - points[i].x_star)) for i in range(len(points) - 1))

    tangent = None
    if x_star is not None and len(points) >= 2:
        dists = [float(np.linalg.norm(p.x_star - x_star)) for p in points]
        i_near = int(np.argmin(dists))
        if 0 < i_near < len(points) - 1:
            d = points[i_near + 1].x_star - points[i_near - 1].x_star
        elif i_near == 0:
            d = points[1].x_star - points[0].x_star
        else:
            d = points[-1].x_star - points[-2].x_star
        norm = float(np.linalg.norm(d))
        if norm > 1e-12:
            tangent = (d / norm).tolist()

    stable_first_coords = [p.x_star[0] for p in points if p.is_stable]
    operating_region_stable = [float(min(stable_first_coords)), float(max(stable_first_coords))] if stable_first_coords else None

    alphas = [p.param_value for p in points]

    return {
        "continuation_parameter": param_name,
        "n_points": len(points),
        "dimension": 1,
        "operating_interval": [float(min(alphas)), float(max(alphas))] if alphas else None,
        "empirically_persistent": len(singular_points) == 0,
        "n_stable": n_stable,
        "n_unstable": n_unstable,
        "n_singular_points": len(singular_points),
        "singular_points": singular_points,
        "max_curvature": float(max(curvatures)) if curvatures else None,
        "mean_curvature": float(np.mean(curvatures)) if curvatures else None,
        "arc_length": arc_length,
        "tangent_at_nearest": tangent,
        "operating_region_stable": operating_region_stable,
    }


def _estimate_contraction_rate(t: np.ndarray, residual: np.ndarray) -> float:
    """
    Fits ln(residual) = ln(A) - rate*t by least squares over the
    portion of the trajectory where the residual is still positive and
    numerically meaningful, returning the fitted rate -- an EFFECTIVE,
    empirical decay-rate estimate (distinct from k_m, the exact,
    provable rate MRC synthesis derives its control law to enforce; see
    control.mrc_synthesis). This estimator is validated to recover a
    known exact rate to machine precision on a synthetic exponential
    decay (see tests/test_explorer_ims_depth.py) before being used on
    any real trajectory. Returns None if fewer than 3 usable points
    remain (e.g. the residual never meaningfully decayed).
    """
    residual = np.asarray(residual)
    t = np.asarray(t)
    mask = residual > 1e-12
    if mask.sum() < 3:
        return None
    coeffs = np.polyfit(t[mask], np.log(residual[mask]), 1)
    return float(-coeffs[0])


def _identify_limiting_state(manifold, x_final: np.ndarray, state_names) -> dict:
    """
    Identifies which single state contributes most to the manifold
    residual at the final trajectory point -- directly addressing a
    real gap: the platform was reporting a recoverability
    classification ("Non-Recoverable") without ANY explanation of which
    part of the system actually caused it, which is not useful for
    engineering decision-making on its own.

    Decomposes the manifold's own nearest-point residual (a weighted
    Euclidean distance -- see ims.manifold.IntrinsicManifold.residual)
    into its per-state components and reports the largest one by name,
    stated as a defensible engineering PROXY, not a rigorous
    decomposition: for a genuinely multi-state coupled system, "which
    state deviates most" and "which state is structurally responsible"
    are related but not identical questions -- a large deviation in one
    state can be a downstream consequence of a different state's own
    root-cause behaviour. This is named honestly in the returned text
    rather than overclaimed as a definitive root-cause analysis.
    """
    projection = manifold.project(x_final)
    diff = np.asarray(x_final) - np.asarray(projection.x_projected)
    abs_diff = np.abs(diff)
    idx = int(np.argmax(abs_diff))
    name = state_names[idx] if idx < len(state_names) else f"x{idx + 1}"

    # Parse a bus/component identifier out of the assembler's own
    # naming convention (see _describe_state) for a more specific
    # engineering-facing description than the bare state name alone.
    if name.startswith("v_"):
        location = f"bus '{name[2:]}'"
    elif name.startswith("i_"):
        location = f"line '{name[2:]}'"
    elif "_em_" in name or "_ctrl_" in name:
        location = f"component '{name.split('_')[0]}'"
    else:
        location = f"state '{name}'"

    return {
        "limiting_state": name,
        "limiting_state_deviation": float(diff[idx]),
        "limiting_state_location": location,
        "note": (
            f"Supplementary diagnostic, not an IMS quantity: the largest single-state deviation from the nearest "
            f"intrinsic-manifold point is in {location} ({name} = {diff[idx]:+.4g} from its manifold-consistent "
            f"value). Offered only as an engineering starting point for investigation -- not the IMS-identified "
            f"cause of recoverability loss, which is a geometric property of the manifold as a whole, not a "
            f"single-state attribution. A large deviation here can also be a downstream consequence of another "
            f"state's behaviour in a coupled network."
        ),
    }


def _recoverability_region_summary(det: dict, boundary_traced: bool) -> dict:
    """
    An always-present summary of the Recoverability Region R and
    Critical Boundary dR (Definition IV.2's positively invariant
    neighborhood U_delta and its boundary), addressing a real gap: this
    content previously only appeared when full directional boundary
    tracing had been explicitly run (Advanced mode), meaning most
    reports -- which use the faster Basic-mode deterministic path --
    never showed these concepts at all.

    Distinguishes clearly between two different things, rather than
    blurring them together: (1) the geometric membership question this
    trajectory's own residual answers directly -- is THIS disturbed
    state inside or outside R, and how far from dR in residual-space
    terms (the recoverability margin, already computed) -- versus (2) a
    full state-space projection of R and dR across a NEIGHBORHOOD of
    disturbances, which requires the more expensive directional
    boundary-tracing computation and is only available when explicitly
    requested.
    """
    margin = det.get("recoverability_margin")
    margin_ratio = det.get("recoverability_margin_ratio")
    tol = det["tolerance_used"]
    inside = margin is not None and margin >= 0

    concept = (
        "The Recoverability Region R is the set of disturbed states whose trajectories return to, and remain "
        "within, an admissible neighborhood of the intrinsic manifold M. Its boundary, the Critical Boundary dR, "
        "separates recoverable initial conditions from non-recoverable ones (Definition IV.2)."
    )

    membership = (
        f"This specific disturbed trajectory's own manifold residual answers the region-membership question "
        f"directly: it settled at a residual of {det['final_residual']:.4g} against an admissible tolerance of "
        f"{tol:.4g}, giving a recoverability margin of {margin:.4g}"
        + (f" ({margin_ratio:.2%} of tolerance remaining)" if inside and margin_ratio is not None else
           f" (tolerance exceeded by a factor of {abs(margin_ratio):.2g})" if margin_ratio is not None else "")
        + f". This means the disturbed state is classified as {'inside' if inside else 'outside'} the "
        f"recoverability region for this specific disturbance -- a direct, single-point answer, not a "
        f"projection across a neighborhood of disturbances."
    )

    if boundary_traced:
        boundary_note = (
            "A full directional boundary trace was also run for this analysis (see below): this projects dR onto "
            "state space by searching outward from the equilibrium along multiple directions, giving an actual "
            "geometric picture of R's extent, not just this one trajectory's membership answer."
        )
    else:
        boundary_note = (
            "A full directional trace of dR was not run for this analysis (this is the faster, default Basic-mode "
            "path). Enabling boundary tracing in Advanced mode computes an actual geometric approximation of R's "
            "extent by searching outward from the equilibrium along multiple directions -- available on request, "
            "not run by default because it requires substantially more simulation than the single-trajectory "
            "assessment above."
        )

    return {
        "concept_explanation": concept,
        "membership_explanation": membership,
        "boundary_note": boundary_note,
        "inside_region": inside,
        "boundary_traced": boundary_traced,
    }


def _resolve_projection_method(requested: str) -> str:
    """
    Resolves a user-requested residual projection method (including
    "auto") into one of IntrinsicManifold.PROJECTION_METHODS.

    "auto" maps directly to "polyline" -- stated honestly as a direct
    mapping to the platform's own currently-recommended default, not an
    adaptive heuristic that switches between methods based on network
    properties. Polyline was chosen as that default only after being
    benchmarked directly against newton_refined (the high-accuracy
    reference) across both a pathological/steep-branch case and a
    well-behaved case: classification matched exactly in every case
    tested, margin agreed within ~4% even at a resolution where
    nearest_sample still misclassified, at the same equilibrium-solve
    cost as nearest_sample (zero extra solves per residual query,
    versus newton_refined's roughly 2x total). If that recommendation
    changes in the future, this is the one place "auto" needs updating.
    """
    if requested in (None, "", "auto"):
        return "polyline"
    if requested not in ("nearest_sample", "polyline", "newton_refined"):
        raise ValueError(f"Unknown residual_projection_method {requested!r}; expected 'auto', 'nearest_sample', 'polyline', or 'newton_refined'.")
    return requested


def _projection_method_validation_summary() -> str:
    """
    A short, report-facing summary of the projection-method benchmark,
    requested to accompany whichever method a report actually used --
    states the general finding (not the full table) so a reader knows
    the chosen method's accuracy has been checked against a reference,
    without needing to re-run the benchmark themselves.
    """
    return (
        "The polyline projection method was validated directly against the newton_refined high-accuracy "
        "reference across representative cases (including a steep/pathological branch and a well-behaved "
        "one, at multiple continuation resolutions): recoverability classification matched exactly in every "
        "case tested, and the recoverability margin agreed to within approximately 4% even at a resolution "
        "where the legacy nearest_sample method still misclassified the result -- while polyline used the "
        "same number of equilibrium solves as nearest_sample (no extra solves per residual query), versus "
        "newton_refined's roughly double the total solves."
    )


def _manifold_residual_explanation() -> str:
    """
    A fixed, general explanation of what the manifold residual r_m
    represents and why its numerical value does not carry a single
    universal physical unit -- requested directly, to accompany the
    deliberate choice not to label the residual axis/value with a
    specific unit (which would be wrong in at least one of this
    platform's two actual formulations).
    """
    return (
        "The manifold residual r_m(x) is the distance from the system's current state x to the nearest point on "
        "the intrinsic manifold M -- the set of states consistent with the system's own equilibrium-tracking "
        "geometry (Definition IV.1/IV.2 in the IMS papers). A small residual means the state is close to this "
        "admissible geometric structure; a residual that shrinks toward zero over time is the geometric signature "
        "of recoverability. Its numerical value does not carry one universal engineering unit because the manifold "
        "is embedded in the full state space, which can mix physically different quantities (e.g. volts, amps, "
        "and controller-internal states with no direct physical unit) into a single distance -- this platform's "
        "generic residual (used for custom networks) is exactly this kind of weighted Euclidean distance across "
        "mixed-unit states. Where a project's manifold is defined by a single analytic formula built entirely "
        "from same-unit quantities (e.g. the network_mrc project's e_m = v_b - (v_o - R*i_l), which is a pure "
        "voltage difference), the residual DOES carry a real physical unit -- but this is a property of that "
        "specific formulation, not something true of the manifold residual in general."
    )


def _geometric_recoverability_narrative(det: dict) -> str:
    """
    A plain-language engineering explanation of WHY a trajectory was
    classified Recoverable / Marginal / Non-Recoverable, built strictly
    from the geometric IMS quantities already computed here (max/final
    residual, contraction rate, recoverability margin, recovery time) --
    not a new metric, and deliberately NOT referencing the supplementary
    limiting-state diagnostic (which is explicitly not an IMS quantity
    and should not be presented as the cause of a geometric result).
    """
    cls = det["classification"]
    max_r, final_r = det["max_residual"], det["final_residual"]
    rate = det.get("contraction_rate")
    margin = det.get("recoverability_margin")
    margin_ratio = det.get("recoverability_margin_ratio")
    tol = det["tolerance_used"]
    recovery_time = det.get("recovery_time")

    # Whether meaningful contraction actually occurred, judged by
    # whether the residual actually dropped substantially -- NOT solely
    # by the fitted rate being nonzero. Confirmed this matters directly:
    # a fitted rate of 0.001 s\u207b\u00b9 technically satisfies "rate > 0" but
    # means the residual would take ~1000s to decay by even 1/e --
    # negligible over any real simulation horizon, and describing that
    # as "contracted... at a rate of 0.001 s\u207b\u00b9" would be misleading.
    contracted_substantially = final_r < max_r * 0.5
    rate_phrase = (
        f"contracted toward the intrinsic manifold at an estimated rate of {rate:.3g} s\u207b\u00b9"
        if rate is not None and rate > 1e-6 and contracted_substantially
        else "did not show meaningful contraction toward the intrinsic manifold within the simulated horizon"
    )

    if cls == "Recoverable":
        return (
            f"The disturbed state began at a manifold residual of {max_r:.4g}. Over the simulated horizon, the "
            f"residual {rate_phrase}"
            + (f", settling within the admissible tolerance of {tol:.4g} by t = {recovery_time:.4g} s" if recovery_time is not None else "")
            + f", reaching a final residual of {final_r:.4g}. This gives a positive recoverability margin of "
            f"{margin:.4g} ({margin_ratio:.2%} of tolerance remaining). Geometrically: the trajectory returned to, "
            f"and remained within, the admissible neighborhood of the intrinsic manifold -- consistent with "
            f"Definition IV.2's recoverability criterion."
        )
    elif cls == "Non-Recoverable":
        return (
            f"The disturbed state began at a manifold residual of {max_r:.4g}. Over the simulated horizon, the "
            f"residual {rate_phrase}, settling at a final residual of {final_r:.4g} -- exceeding the admissible "
            f"tolerance of {tol:.4g}. This gives a negative recoverability margin of {margin:.4g} "
            f"(tolerance exceeded by a factor of {abs(margin_ratio):.2g}). Geometrically: the trajectory did not "
            f"return to, or remain within, the admissible neighborhood of the intrinsic manifold within the "
            f"simulated horizon -- Definition IV.2's recoverability criterion is not satisfied here."
        )
    else:  # Marginal
        return (
            f"The disturbed state began at a manifold residual of {max_r:.4g}. Over the simulated horizon, the "
            f"residual {rate_phrase}, settling at a final residual of {final_r:.4g} -- close to, but not "
            f"comfortably within, the admissible tolerance of {tol:.4g} (recoverability margin {margin:.4g}). "
            f"Geometrically: the trajectory is near the boundary between recoverable and non-recoverable behaviour; "
            f"a longer simulated horizon or a tighter/looser tolerance could change this classification, so this "
            f"result should be treated as borderline rather than conclusive."
        )


def _deterministic_recoverability_assessment(t, residual, tolerance: float = None) -> dict:
    """
    Classifies recoverability directly from the manifold-residual
    trajectory e_m(t), per Definition IV.2 in the IMS papers: a
    trajectory is recoverable if it stays within a bounded
    neighborhood and e_m(t) -> 0. This is the PRIMARY recoverability
    signal in this framework -- deterministic, computed once per
    disturbance from the actual simulated trajectory, not a statistical
    estimate over many random samples (Monte Carlo remains available
    separately, as a validation check ON this deterministic result, not
    a replacement for it).

    Validated against two known-contrasting cases from the network_mrc
    project (see tests/test_project_builder.py and the paper's own
    Fig. 4 ablation): with the manifold-reshaping contraction gain
    active (km=500), the residual decays from -5.0 to ~-6.6e-6 in one
    horizon -- classified Recoverable here. With the gain reduced to
    near zero (km=0.001, mirroring the paper's "MRC-no-IMS" ablation),
    the residual barely moves (-5.0 -> -4.9997) -- classified
    Non-Recoverable here. Both confirmed to produce the expected label
    before this function was used anywhere else.

    Metrics returned match the quantities the papers themselves report
    (peak/integrated manifold residual, final residual, recovery time):
    max/RMS/integral of |e_m|, final e_m, recovery time (first time
    |e_m| drops below and STAYS below tolerance), and an empirical
    contraction-rate estimate (see _estimate_contraction_rate).
    """
    residual = np.asarray(residual, dtype=float)
    t = np.asarray(t, dtype=float)
    abs_r = np.abs(residual)
    max_r = float(np.max(abs_r))
    rms_r = float(np.sqrt(np.mean(residual ** 2)))
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    integral_r = float(trapz_fn(abs_r, t))
    final_r = float(abs_r[-1])

    if tolerance is None:
        tolerance = max(1e-6, 0.02 * max_r)  # 2% of the peak deviation, or a small absolute floor

    recovery_time = None
    for i in range(len(abs_r)):
        if np.all(abs_r[i:] <= tolerance):
            recovery_time = float(t[i])
            break

    contraction_rate = _estimate_contraction_rate(t, abs_r)

    if final_r <= tolerance and recovery_time is not None:
        classification = "Recoverable"
    elif final_r <= tolerance * 25:
        classification = "Marginal"
    else:
        classification = "Non-Recoverable"

    # Recoverability margin: signed distance, in residual space, between
    # the admissible tolerance and where the trajectory actually settled
    # -- positive means margin remains (inside the admissible
    # neighborhood with room to spare), negative means the tolerance was
    # exceeded (outside). This is the geometric quantity Definition IV.2
    # is actually stated in terms of (the tolerance delta the
    # neighborhood U_delta is built from), not a separate heuristic.
    margin = float(tolerance - final_r)
    margin_ratio = float(margin / tolerance) if tolerance > 0 else None

    return {
        "max_residual": max_r, "rms_residual": rms_r, "integral_abs_residual": integral_r,
        "final_residual": final_r, "recovery_time": recovery_time,
        "contraction_rate": contraction_rate, "tolerance_used": float(tolerance),
        "recoverability_margin": margin, "recoverability_margin_ratio": margin_ratio,
        "classification": classification,
    }


def _check_ims_conditions(eigenvalues, contraction_rate: float, final_residual: float, tolerance: float,
                          transverse_target_rate: float = None) -> dict:
    """
    Tri-state (SATISFIED / VIOLATED / NOT_ESTABLISHED) audit against
    Definition IV.3's IMS conditions. Deliberately keeps five distinct
    mathematical claims separate and never infers one from another:

      1. full_system_local_stability  -- Lyapunov's indirect method: all
         EQUILIBRIUM Jacobian eigenvalues have negative real part.
         Computable whenever `eigenvalues` is available.
      2. normal_hyperbolicity         -- no equilibrium eigenvalue lies
         ON the imaginary axis (the condition Lyapunov's indirect method
         itself needs to be conclusive at all). This is about the
         full-system equilibrium, NOT a transverse/tangential
         decomposition on the manifold -- that is claim 3 below, a
         separate and strictly harder question.
      3. transverse_contraction       -- analytic: requires knowing
         WHICH eigenvalue is transverse, which requires the manifold's
         own target contraction rate k_m (`transverse_target_rate`).
         When given, the eigenvalue closest to -k_m is treated as
         transverse (mirrors mrc_designer.auto_manifold's own
         convention): SATISFIED iff it actually matches -k_m, VIOLATED
         otherwise. Without `transverse_target_rate` this is honestly
         NOT_ESTABLISHED -- a numerical eigenvalue list alone does not
         identify which eigenvalue is transverse.
      4. reduced_dynamics_stable      -- analytic tangential/reduced
         stability, using the SAME transverse/tangential split as claim
         3 (so it is NOT_ESTABLISHED whenever claim 3 is, regardless of
         what any empirical simulation shows -- reduced dynamics
         stability is never inferred from full-system stability or from
         empirical recovery).
      5. empirical_finite_horizon_recovery -- a separate, weaker,
         numerical-simulation observation: whether the observed residual
         trajectory contracted at a positive, significant fitted rate
         and settled below tolerance within the simulated horizon. This
         is evidence, not a proof of claim 3 or 4, and is reported
         independently rather than folded into `overall_status`.
      6. certified_regional_ims       -- never established by this
         runtime check (Theorem IV.1's regional guarantee needs
         analytical work this function does not attempt) -- always
         NOT_ESTABLISHED here.

    `overall_status` is SATISFIED only when claims 1-4 are ALL
    SATISFIED (i.e. only when an analytic transverse/tangential split
    is actually available and both halves hold); VIOLATED if any of
    1-4 is VIOLATED; otherwise NOT_ESTABLISHED. NOT_ESTABLISHED is the
    expected, honest result for the generic structure-agnostic IMS
    Analysis pathway, which has no analytic tangent space and so never
    receives a `transverse_target_rate` -- it is not a "failure".
    """
    eig_list = list(eigenvalues) if eigenvalues is not None else []
    eig_real = [e.real if hasattr(e, "real") else float(e) for e in eig_list]
    eig_imag = [e.imag if hasattr(e, "imag") else 0.0 for e in eig_list]

    if not eig_list:
        full_system_local_stability = "NOT_ESTABLISHED"
        normal_hyperbolicity = "NOT_ESTABLISHED"
    else:
        # Scale-aware "on the imaginary axis" tolerance: eigenvalues here
        # routinely come from a finite-difference Jacobian, so an
        # analytically-exact zero mode (a real, structural property some
        # reference models are documented to have) typically shows up as a
        # small but nonzero residual (~1e-7-1e-5) rather than exactly 0. A
        # fixed 1e-8 absolute tolerance would misclassify that noise as
        # "off axis" and hide a genuine zero eigenvalue; scaling the
        # tolerance to the spectrum's own magnitude catches it without
        # loosening the check for genuinely well-separated spectra.
        axis_tol = max(1e-8, 1e-6 * max((abs(re) for re in eig_real), default=1.0))
        full_system_local_stability = "SATISFIED" if all(re < -axis_tol for re in eig_real) else "VIOLATED"
        normal_hyperbolicity = "SATISFIED" if all(abs(re) > axis_tol for re in eig_real) else "VIOLATED"

    transverse_contraction = "NOT_ESTABLISHED"
    reduced_dynamics_stable = "NOT_ESTABLISHED"
    transverse_eigenvalue = None
    tangential_eigenvalues = None
    if transverse_target_rate is not None and eig_list:
        km = float(transverse_target_rate)
        idx = int(np.argmin([abs(re - (-km)) for re in eig_real]))
        transverse_eigenvalue = {"re": eig_real[idx], "im": eig_imag[idx]}
        matches = abs(eig_real[idx] - (-km)) < max(1e-2 * km, 1.0) and abs(eig_imag[idx]) < 1.0
        transverse_contraction = "SATISFIED" if matches else "VIOLATED"
        tangential = [(re, im) for i, (re, im) in enumerate(zip(eig_real, eig_imag)) if i != idx]
        tangential_eigenvalues = [{"re": re, "im": im} for re, im in tangential]
        reduced_dynamics_stable = (
            ("SATISFIED" if all(re < -1e-8 for re, _ in tangential) else "VIOLATED")
            if tangential else "NOT_ESTABLISHED"
        )

    empirical_finite_horizon_recovery = bool(
        contraction_rate is not None and contraction_rate > 1e-3
        and final_residual is not None and tolerance is not None and final_residual <= tolerance
    )

    certified_regional_ims = "NOT_ESTABLISHED"

    required = [full_system_local_stability, normal_hyperbolicity, transverse_contraction, reduced_dynamics_stable]
    if any(s == "VIOLATED" for s in required):
        overall_status = "VIOLATED"
    elif all(s == "SATISFIED" for s in required):
        overall_status = "SATISFIED"
    else:
        overall_status = "NOT_ESTABLISHED"

    return {
        "full_system_local_stability": full_system_local_stability,
        "normal_hyperbolicity": normal_hyperbolicity,
        "transverse_contraction": transverse_contraction,
        "transverse_eigenvalue": transverse_eigenvalue,
        "reduced_dynamics_stable": reduced_dynamics_stable,
        "tangential_eigenvalues": tangential_eigenvalues,
        "empirical_finite_horizon_recovery": empirical_finite_horizon_recovery,
        "certified_regional_ims": certified_regional_ims,
        "overall_status": overall_status,
    }



def stage_build(project: dict, payload: dict) -> dict:
    system, param_name, extra = _build_system(project, payload)

    if project["category"] in ("converter_topology", "custom_network") and hasattr(system, "network"):
        network_summary = _network_summary_from_network(system.network)
        component_chain = _component_chain_from_network(system.network)
        topology_tree = _topology_tree_from_network(system.network)
    else:
        network_summary = project.get("network_summary", {})
        component_chain = project.get("component_chain", [])
        topology_tree = []

    n_inputs = len(system.input_names)
    result = {
        "state_names": list(system.state_names),
        "input_names": list(system.input_names),
        "param_name_for_continuation": param_name,
        "params_used": dict(system.params),
        "category": project["category"],
        "network_summary": network_summary,
        "component_chain": component_chain,
        "topology_tree": topology_tree,
        "model_order": {"n_states": system.n_states, "n_inputs": n_inputs, "n_outputs": system.n_states},
        "controller_description": _controller_description(project, getattr(system, "network", None)),
        **extra,
    }
    if project.get("governing_equations"):
        result["governing_equations"] = project["governing_equations"]
    if project.get("engineering_outputs"):
        result["engineering_outputs"] = project["engineering_outputs"]
    if hasattr(system, "delta_max_frac_pi"):
        result["admissible_envelope"] = {
            "delta_max": system.delta_max_frac_pi * np.pi,
            "omega_max": system.omega_max,
            "description": f"|\u03b4| < {system.delta_max_frac_pi}\u00b7\u03c0 rad, |\u03c9| < {system.omega_max} rad/s",
        }
    if project["category"] == "custom_network":
        result["default_input"] = project.get("default_input", [0.5])
    return result


def _component_operating_points(network: Network, x_star: np.ndarray, state_names: list) -> dict:
    """
    Per-component operating point at a given equilibrium: bus voltage,
    terminal (port) current, and electrical power. Reuses
    `Converter.current_injection` directly -- the exact method the
    Automatic Model Builder itself uses to assemble the network's KCL
    equations (network/assembler.py) -- rather than a new formula, so
    the reported current is provably the same current the simulation
    itself uses, correct for whichever topology-specific relation that
    component's ElectricalModel implements (e.g. i_L for a buck
    converter, (1-D)*i_L for a boost converter).
    """
    points = {}
    for comp in network.components:
        if not isinstance(comp, Converter):
            continue
        v_name = f"v_{comp.bus}"
        if v_name not in state_names:
            continue
        v_bus = float(x_star[state_names.index(v_name)])
        prefix = f"{comp.id}_"
        local_indices = [i for i, n in enumerate(state_names) if n.startswith(prefix)]
        local_x = x_star[local_indices] if local_indices else np.zeros(0)
        i_port = float(comp.current_injection(v_bus, local_x, None))
        points[comp.id] = {
            "type": type(comp.electrical_model).__name__,
            "bus": comp.bus,
            "operating_voltage": v_bus,
            "operating_current": i_port,
            "operating_power": v_bus * i_port,
        }
    return points


def _describe_state(name: str) -> str:
    """
    Human-readable description of a state's own governing equation,
    inferred from the Automatic Model Builder's own naming convention
    (network.assembler): "v_{bus}" for a bus voltage (KCL), "i_{line}"
    for a line current (KVL), "{comp}_em_..." for a converter's own
    electrical-model state, "{comp}_ctrl_..." for a controller's
    internal state (e.g. a PI integrator). Used to label per-equation
    residuals with something more useful than a bare index -- the
    review's own example ("Power balance bus 1", "Line current
    equation", ...) is exactly this idea; the labels here are generated
    from the real state name rather than a hand-maintained list, so
    they can't drift out of sync with what the assembler actually built.
    """
    if name.startswith("v_"):
        return f"KCL (power/current balance) at bus '{name[2:]}'"
    if name.startswith("i_"):
        return f"KVL (branch current) on line '{name[2:]}'"
    if "_ctrl_" in name:
        comp, _, state = name.partition("_ctrl_")
        return f"controller internal state '{state}' on component '{comp}'"
    if "_em_" in name:
        comp, _, state = name.partition("_em_")
        return f"electrical-model state '{state}' on component '{comp}'"
    return f"state '{name}'"


def _structural_rank_check(system, x_guess: np.ndarray, u: np.ndarray) -> None:
    """
    Performs the structural analysis the review asked be done BEFORE
    calling the solver, not only after it fails: evaluates the Jacobian
    at the INITIAL GUESS and checks its rank against the number of
    states. Note on architecture, stated plainly because it changes what
    "under-constrained" can mean here: this platform's dx/dt = f(x,u) is
    always exactly square by construction (network.assembler.dynamics
    allocates exactly one equation slot per state; there is no separate
    algebraic-constraint layer where an equation could go missing the
    way it can in a general DAE). So a rank deficiency here always means
    numerical/structural degeneracy of the Jacobian -- e.g. a state
    whose own dynamics don't depend on anything that, in turn, depends
    on it -- not a literal missing equation. Rank is checked with a
    relative tolerance because the ABSOLUTE magnitudes here can differ
    by many orders across state types (volts vs. amps vs. controller
    integral states), which would make a fixed absolute tolerance
    meaningless.
    """
    n = system.n_states
    J = system.jacobian(x_guess, u)
    # Row-normalise before checking rank: different state types (volts,
    # amps, controller integral states) routinely have Jacobian entries
    # that differ by many orders of magnitude even in a perfectly
    # well-posed system (confirmed directly: an extreme but physically
    # valid inductance value, 1e-80 H, produces a row with ~1e80-scale
    # entries that would swamp numpy's default rank tolerance and
    # falsely flag an otherwise nonsingular system -- L doesn't even
    # appear in the equilibrium condition itself, only the transient).
    # Row-normalising first removes that scale artifact while still
    # catching a GENUINE structural degeneracy (confirmed: the PI
    # controller's marginal integrator mode still shows a rank deficit
    # after normalising, since that one is real, not a scaling effect).
    row_norms = np.linalg.norm(J, axis=1, keepdims=True)
    row_norms_safe = np.where(row_norms > 0, row_norms, 1.0)
    J_normalized = J / row_norms_safe
    rank = int(np.linalg.matrix_rank(J_normalized, tol=1e-8))
    if rank < n:
        # Identify likely-decoupled states two ways: rows of J that are
        # exactly zero (a state whose own equation doesn't depend on
        # anything), AND columns that are exactly zero (a state that
        # has no influence on any equation, including its own) --
        # confirmed directly that the second pattern is what actually
        # occurs for a controller's integral state when the duty ratio
        # is saturated (clamp_duty) at the initial guess: the control
        # signal is pinned to its floor/ceiling regardless of small
        # changes in the integral state, so its Jacobian COLUMN is
        # zero there even though the state is real and matters once
        # the duty ratio is back in range.
        col_norms = np.linalg.norm(J, axis=0)
        zero_rows = [system.state_names[i] for i in range(n) if row_norms[i, 0] == 0]
        zero_cols = [system.state_names[i] for i in range(n) if col_norms[i] == 0]
        ctrl_zero_cols = [s for s in zero_cols if "_ctrl_" in s]
        extra_note = ""
        if ctrl_zero_cols:
            extra_note = (
                f" Note: {ctrl_zero_cols} showed zero influence on every equation AT THIS INITIAL GUESS "
                f"specifically -- this commonly happens when a PI controller's duty ratio is saturated "
                f"(clamped to its floor/ceiling) at the guess point, making the integral state locally "
                f"invisible to the Jacobian even though it is a real state. A bare integral controller "
                f"(no other reference-tracking constraint) is also known to make Newton-based equilibrium "
                f"solving unreliable in general, independent of this specific guess -- confirmed directly: "
                f"even bypassing this check, the underlying solver does not reliably converge for this "
                f"controller either. The constant-duty controller is validated to solve reliably; if a "
                f"regulated (PI) equilibrium is genuinely needed, this is a known, real limitation of the "
                f"current version, not something a better initial guess alone is likely to fix."
            )
        raise ValueError(
            f"structural check failed before attempting the equilibrium solve: "
            f"rank(Jacobian at initial guess) = {rank}, expected {n} (one per state). "
            f"States: {list(system.state_names)}. "
            + (f"State(s) with an apparently-decoupled equation (zero Jacobian row): {zero_rows}. "
               if zero_rows else "") +
            (f"State(s) with no influence on any equation (zero Jacobian column): {zero_cols}. "
             if zero_cols else "") +
            f"This is a structural property of the network as built, not a solver convergence issue -- "
            f"the equilibrium solver was not called."
            + extra_note
        )


def _try_pi_regulated_equilibrium(system, network: Network, state_guess: np.ndarray, u: np.ndarray):
    """
    A specialised equilibrium solve for a PI-regulated converter,
    exploiting a validated mathematical property rather than a plain
    Newton search over all states jointly (which is genuinely
    ill-conditioned here -- confirmed directly: e_int shifts the
    converter's electrical state by orders of magnitude per unit,
    causing joint Newton search to either fail outright or converge to
    a self-consistent-looking but wrong point).

    The property: at any true equilibrium, d(e_int)/dt = 0 forces the
    regulated bus voltage to equal the controller's v_nom EXACTLY (not
    something to search for -- it's a direct consequence of the
    controller's own equation). Fixing that state at v_nom and solving
    the converter's own electrical-model equation (e.g. inductor
    current) for e_int via a well-conditioned 1D root-find, nested
    inside an outer search over the remaining states, avoids the
    ill-conditioning entirely. Validated to converge to machine
    precision (residual ~1e-11) on both a single-bus case and a
    two-bus, line-connected case matching an actual reported network,
    confirmed independently against KCL at every bus in both cases (see
    tests/test_project_builder.py).

    Restricted to exactly one PI-regulated converter with exactly one
    electrical-model state (true for Buck/Boost/BuckBoost as currently
    implemented) -- returns None for anything else, leaving the caller
    to fall back to the structural-check's own diagnostic message
    rather than silently extending an unvalidated method to a case it
    hasn't been checked against.
    """
    from scipy.optimize import brentq, fsolve

    state_names = list(system.state_names)
    ctrl_candidates = [i for i, n in enumerate(state_names) if n.endswith("_ctrl_e_int")]
    if len(ctrl_candidates) != 1:
        return None
    ctrl_idx = ctrl_candidates[0]
    comp_id = state_names[ctrl_idx].split("_ctrl_")[0]

    conv = next((c for c in network.components if getattr(c, "id", None) == comp_id and isinstance(c, Converter)), None)
    if conv is None or not isinstance(conv.controller, (SimplePIVoltageController, PIDutyController)):
        return None
    # SimplePIVoltageController.control_signal/local_dynamics both use
    # "p['v_nom'] if u is None else u" -- i.e. the exogenous input u
    # OVERRIDES the controller's own fixed v_nom parameter whenever
    # it's provided. This matters here specifically: a manifold sweep
    # varies u deliberately (see IntrinsicManifold's default input
    # setter, u[0]=alpha), so using the FIXED params['v_nom'] instead of
    # the ACTUAL effective v_nom would silently find the same
    # equilibrium at every sweep point rather than tracing a real
    # manifold -- confirmed this exact mismatch by checking the real
    # controller's own source before trusting either value.
    v_nom = float(u[0]) if len(u) > 0 else conv.controller.params.get("v_nom")
    if v_nom is None:
        return None

    v_bus_name = f"v_{conv.bus}"
    if v_bus_name not in state_names:
        return None
    v_bus_idx = state_names.index(v_bus_name)

    em_indices = [i for i, n in enumerate(state_names) if n.startswith(f"{comp_id}_em_")]
    if len(em_indices) != 1:
        return None
    em_idx = em_indices[0]

    n = len(state_names)
    free_idx = [i for i in range(n) if i not in (v_bus_idx, ctrl_idx)]
    # Equations kept for the OUTER solve: exclude the controller's own
    # equation (auto-satisfied once v_bus=v_nom) and the electrical
    # model's own equation (consumed by the inner solve, which finds
    # e_int specifically to zero it out -- including it again here
    # would hand the outer solve an equation that's identically ~0 for
    # every x_free by construction, making that Jacobian row
    # structurally near-zero and causing exactly the "not making good
    # progress" failure confirmed directly while debugging this).
    # v_bus_idx's OWN equation (its KCL) is correctly KEPT even though
    # its VALUE is fixed -- fixing the value doesn't automatically
    # satisfy its equation; that's a real constraint on the other
    # states.
    outer_eq_idx = [i for i in range(n) if i not in (ctrl_idx, em_idx)]

    def inner_solve_ctrl(x_partial, bracket=(-1e5, 1e5)):
        def f(e_int):
            x = x_partial.copy()
            x[ctrl_idx] = e_int
            return system.dynamics(0.0, x, u, system.params)[em_idx]
        try:
            return brentq(f, *bracket, xtol=1e-12)
        except ValueError:
            return None

    def outer(x_free):
        x = np.zeros(n)
        x[v_bus_idx] = v_nom
        x[free_idx] = x_free
        e_int = inner_solve_ctrl(x)
        if e_int is None:
            return np.full(len(outer_eq_idx), 1e6)
        x[ctrl_idx] = e_int
        return system.dynamics(0.0, x, u, system.params)[outer_eq_idx]

    x0_free = np.array([state_guess[i] for i in free_idx])
    sol, info, ier, msg = fsolve(outer, x0_free, full_output=True)
    x_full = np.zeros(n)
    x_full[v_bus_idx] = v_nom
    x_full[free_idx] = sol
    e_int_final = inner_solve_ctrl(x_full)
    residual_norm = None
    if e_int_final is not None:
        x_full[ctrl_idx] = e_int_final
        residual_norm = float(np.linalg.norm(system.dynamics(0.0, x_full, u, system.params)))

    if residual_norm is None or residual_norm > 1e-6:
        # The exact guess given can fail here (confirmed directly: an
        # exactly-zero current guess -- which the frontend's own
        # defaultStateGuess() produces for every line current -- can
        # make the outer fsolve fail to converge, even though the same
        # network converges cleanly from a nearby, non-degenerate
        # guess). Retry once from a perturbed guess (any exactly-zero
        # current-like free state nudged to a modest nonzero value)
        # before giving up, rather than being fragile to this one
        # common starting point.
        x0_retry = x0_free.copy()
        for k, idx in enumerate(free_idx):
            if state_names[idx].startswith("i_") and x0_retry[k] == 0.0:
                x0_retry[k] = 1.0
        if not np.allclose(x0_retry, x0_free):
            sol, info, ier, msg = fsolve(outer, x0_retry, full_output=True)
            x_full = np.zeros(n)
            x_full[v_bus_idx] = v_nom
            x_full[free_idx] = sol
            e_int_final = inner_solve_ctrl(x_full)
            if e_int_final is not None:
                x_full[ctrl_idx] = e_int_final
                residual_norm = float(np.linalg.norm(system.dynamics(0.0, x_full, u, system.params)))
            else:
                residual_norm = None

    if residual_norm is None or residual_norm > 1e-6:
        return None
    return x_full


def stage_equilibrium(project: dict, payload: dict) -> dict:
    system, param_name, extra = _build_system(project, payload)

    if project["category"] == "converter_topology":
        v_guess = extra["v_guess"]
        state_guess = np.array([v_guess, v_guess / 10.0]) if payload.get("state_guess") is None else np.array(payload["state_guess"])
        u = np.array(payload.get("nominal_input", project["default_input"]), dtype=float)
    else:
        state_guess = np.array(payload["state_guess"], dtype=float)
        u = np.array(payload.get("nominal_input", project["default_input"]), dtype=float)

    pi_x_star = None
    if hasattr(system, "network"):
        pi_x_star = _try_pi_regulated_equilibrium(system, system.network, state_guess, u)

    if pi_x_star is not None:
        # The validated nested solve succeeded -- use its result directly,
        # computing the standard diagnostics (eigenvalues, Jacobian, etc.)
        # at this point exactly as find_equilibrium would, rather than
        # re-deriving a separate result type.
        t0 = time.perf_counter()
        J_pi = system.jacobian(pi_x_star, u)
        eigs_pi = np.linalg.eigvals(J_pi)
        try:
            cond_pi_val = np.linalg.cond(J_pi)
            cond_pi = float(cond_pi_val) if np.isfinite(cond_pi_val) else None
        except np.linalg.LinAlgError:
            cond_pi = None
        residual_pi = float(np.linalg.norm(system.dynamics(0.0, pi_x_star, u, system.params)))
        eq = EquilibriumResult(
            x_star=pi_x_star, converged=True, residual_norm=residual_pi, eigenvalues=eigs_pi,
            n_function_evals=None, jacobian=J_pi, jacobian_condition_number=cond_pi,
            computation_time_s=time.perf_counter() - t0,
        )
    else:
        _structural_rank_check(system, state_guess, u)
        eq = system.find_equilibrium(state_guess, u=u)
    scale = max(1.0, float(np.linalg.norm(eq.x_star)))
    # Two checks, not one: the relative check alone is defeated exactly
    # when it matters most -- if the solver has diverged (e.g. a PI
    # controller's integrator state, which is only weakly constrained
    # at equilibrium and can run away to a huge value; see
    # tests/test_project_builder.py), the scale inflates right along
    # with the divergence, making a genuinely bad absolute residual look
    # small in relative terms. An absolute ceiling catches this case
    # even when the relative one doesn't.
    ok = (eq.residual_norm / scale < 1e-6) and (eq.residual_norm < 1.0)
    if not ok:
        # Enriched diagnostics on failure, not just the aggregate norm:
        # the per-equation residual vector (which single equation is
        # actually bad), whether the FOUND point contains any non-finite
        # value (confirms/rules out an overflow-driven divergence, e.g.
        # an under-constrained island with no finite equilibrium -- see
        # _check_network_connectivity, which now catches the specific
        # case that motivated this), and the Jacobian condition number
        # (previously only surfaced on success).
        F_at_xstar = system.dynamics(0.0, eq.x_star, u, system.params)
        per_eq = [float(v) if np.isfinite(v) else None for v in F_at_xstar]
        labeled_residuals = {system.state_names[i]: {"description": _describe_state(system.state_names[i]), "residual": per_eq[i]}
                             for i in range(len(per_eq))}
        has_nonfinite_state = bool(np.any(~np.isfinite(eq.x_star)))
        cond = None
        try:
            J_fail = system.jacobian(eq.x_star, u)
            cond_val = np.linalg.cond(J_fail)
            cond = float(cond_val) if np.isfinite(cond_val) else None
        except Exception:
            pass
        raise ValueError(
            f"equilibrium did not converge to adequate precision "
            f"(residual={eq.residual_norm:.3e}, residual/scale={eq.residual_norm / scale:.2e}). "
            f"Per-equation residual at the last iterate: {labeled_residuals}. "
            f"Non-finite state reached: {has_nonfinite_state}. "
            f"Jacobian condition number there: {cond if cond is not None else 'singular/undefined'}. "
            f"A non-finite or astronomically large state most often means an under-constrained network "
            f"(e.g. an island of buses with only loads and no source -- usually caught before this point, "
            f"see the Build Model stage's own error -- or a genuinely missing connection)."
        )

    jacobian_rank = None
    jacobian_spectral_radius = None
    if eq.jacobian is not None:
        # Same row-normalisation as _structural_rank_check, for
        # consistency: different state types (volts, amps, controller
        # integral states) can have Jacobian entries differing by many
        # orders of magnitude even in a well-posed system, which would
        # otherwise swamp numpy's default rank tolerance.
        J_row_norms = np.linalg.norm(eq.jacobian, axis=1, keepdims=True)
        J_row_norms_safe = np.where(J_row_norms > 0, J_row_norms, 1.0)
        jacobian_rank = int(np.linalg.matrix_rank(eq.jacobian / J_row_norms_safe, tol=1e-8))
    if eq.eigenvalues is not None and len(eq.eigenvalues) > 0:
        jacobian_spectral_radius = float(np.max(np.abs(eq.eigenvalues)))

    result = {
        "state_names": list(system.state_names),
        "x_star": eq.x_star.tolist(),
        "stable": bool(eq.is_stable),
        "is_hyperbolic": bool(eq.is_hyperbolic),
        "eigenvalues": [{"re": float(e.real), "im": float(e.imag)} for e in eq.eigenvalues],
        "residual_norm": float(eq.residual_norm),
        "nominal_input": u.tolist(),
        "jacobian": eq.jacobian.tolist() if eq.jacobian is not None else None,
        "jacobian_rank": jacobian_rank,
        "jacobian_spectral_radius": jacobian_spectral_radius,
        "solver_diagnostics": {
            "function_evaluations": eq.n_function_evals,
            "jacobian_condition_number": eq.jacobian_condition_number,
            "computation_time_s": eq.computation_time_s,
        },
    }
    if hasattr(system, "network"):
        result["component_operating_points"] = _component_operating_points(system.network, eq.x_star, list(system.state_names))
    if project.get("model_key") == "grid_forming_inverter":
        delta_star = float(eq.x_star[0])
        p_e = float(system.electrical_power(delta_star))
        power_mismatch = p_e - float(u[0])
        result["electrical_operating_point"] = {
            "delta": delta_star, "electrical_power": p_e, "power_setpoint": float(u[0]),
            "power_mismatch": power_mismatch,
            "matches_setpoint": abs(power_mismatch) < 1e-6,
        }
        result["transverse_stability"] = {
            "is_hyperbolic": bool(eq.is_hyperbolic),
            "description": (
                "The operating point satisfies the normal hyperbolicity condition (no eigenvalue has zero real part), "
                "indicating the intrinsic manifold remains structurally stable under small perturbations at this point."
            ) if eq.is_hyperbolic else (
                "The operating point does NOT satisfy the normal hyperbolicity condition -- at least one eigenvalue has "
                "(near-)zero real part, indicating the intrinsic manifold's structure may be degenerate here (e.g. near a fold)."
            ),
        }
        result["characteristic_time_constants"] = [
            (1.0 / abs(e["re"])) if abs(e["re"]) > 1e-12 else None for e in result["eigenvalues"]
        ]
    return result


def stage_simulate(project: dict, payload: dict) -> dict:
    system, param_name, extra = _build_system(project, payload)
    u = np.array(payload["nominal_input"], dtype=float)
    x_star = np.array(payload["x_star"], dtype=float)
    dist = payload.get("disturbance", {"type": "offset", "value": [0.0] * system.n_states})
    disturbed = x_star + np.array(dist["value"], dtype=float) if dist["type"] == "offset" else np.array(dist["value"], dtype=float)
    horizon = float(payload.get("horizon", project.get("default_horizon", 5.0)))

    sim = Simulator(system, method="RK45")
    traj_ol = sim.simulate(disturbed, (0.0, horizon), u=u, n_eval=300)
    ol_metrics = _performance_metrics(np.array(traj_ol.t), np.array(traj_ol.x), x_star)
    result = {
        "disturbed_state": disturbed.tolist(),
        "trajectory_open_loop": {"t": traj_ol.t.tolist(), "x": traj_ol.x.tolist(), "success": traj_ol.success},
        "performance_metrics_open_loop": ol_metrics,
        "controller_info": {
            "type": _controller_description(project, getattr(system, "network", None))
            if project["category"] == "custom_network" else "None (open loop)"
        },
    }
    if project.get("model_key") == "grid_forming_inverter":
        # Electrical power P_e(t) = (E*V/X)*sin(delta(t)), via the
        # model's own validated electrical_power() method -- not a
        # separately-maintained formula that could drift from the
        # actual dynamics.
        p_e_ol = [float(system.electrical_power(traj_ol.x[0, k])) for k in range(traj_ol.x.shape[1])]
        result["electrical_power_open_loop"] = {"t": traj_ol.t.tolist(), "power": p_e_ol}
        pe_metrics_ol = _performance_metrics(np.array(traj_ol.t), np.array([p_e_ol]), np.array([float(u[0])]))
        result["electrical_power_metrics_open_loop"] = {k: v[0] for k, v in pe_metrics_ol.items()}

    if payload.get("mrc_enabled") and project["category"] == "single_model":
        M = IntrinsicManifold(system, param_name=param_name).build(
            alpha_range=tuple(project["sweep_range"]), n_points=project["sweep_points"],
            x0_guess=x_star, keep_unstable=True,
        )
        Q = np.diag(payload.get("Q_diag", project["q_default"]))
        R = np.diag(payload.get("R_diag", project["r_default"]))
        u_min = np.array(payload.get("u_min", project.get("u_min_default"))) if payload.get("u_min") or project.get("u_min_default") else None
        u_max = np.array(payload.get("u_max", project.get("u_max_default"))) if payload.get("u_max") or project.get("u_max_default") else None
        lqr = ScheduledLQRControl(system, M, Q=Q, R=R, u_min=u_min, u_max=u_max)
        lqr.precompute_gain_schedule(u_nominal=u)
        controller = lqr.as_controller(u_nominal=u)
        traj_cl = sim.simulate(disturbed, (0.0, horizon), controller=controller, n_eval=300)
        result["trajectory_closed_loop"] = {"t": traj_cl.t.tolist(), "x": traj_cl.x.tolist(), "success": traj_cl.success}
        result["performance_metrics_closed_loop"] = _performance_metrics(np.array(traj_cl.t), np.array(traj_cl.x), x_star)
        if project.get("model_key") == "grid_forming_inverter":
            p_e_cl = [float(system.electrical_power(traj_cl.x[0, k])) for k in range(traj_cl.x.shape[1])]
            result["electrical_power_closed_loop"] = {"t": traj_cl.t.tolist(), "power": p_e_cl}
            pe_metrics_cl = _performance_metrics(np.array(traj_cl.t), np.array([p_e_cl]), np.array([float(u[0])]))
            result["electrical_power_metrics_closed_loop"] = {k: v[0] for k, v in pe_metrics_cl.items()}

        # Control effort: reconstruct u(t) along the closed-loop
        # trajectory by re-evaluating the same controller function at
        # each sampled point (the Simulator itself does not record
        # control history, so this is a faithful post-hoc replay, not
        # an approximation).
        u_hist = np.array([controller(traj_cl.t[k], traj_cl.x[:, k]) for k in range(traj_cl.x.shape[1])])
        result["control_effort"] = {
            "rms": float(np.sqrt(np.mean(u_hist ** 2))),
            "peak": float(np.max(np.abs(u_hist))),
        }

        # Contraction-rate estimate: the manifold residual along the
        # closed-loop trajectory, fitted to an effective exponential
        # decay rate (validated against a known synthetic case -- see
        # _estimate_contraction_rate's docstring). This is an empirical
        # estimate for a Scheduled LQR controller, not a provable
        # guarantee the way k_m is for IMS-native MRC.
        residual_cl = M.residual_trajectory(traj_cl.x)
        rate = _estimate_contraction_rate(traj_cl.t, residual_cl)
        result["manifold_residual_trajectory_closed_loop"] = {"t": traj_cl.t.tolist(), "residual": residual_cl.tolist()}
        result["controller_info"] = {
            "type": "Scheduled LQR (Conventional Control Library)",
            "note": "Not IMS-native MRC -- see the Network + Auto-MRC project for genuine Symbolic-Engine-derived MRC.",
            "Q_diag": Q.diagonal().tolist(), "R_diag": R.diagonal().tolist(),
            "estimated_contraction_rate": rate,
        }

    return result


def _recoverability_extras(report, x_star: np.ndarray) -> dict:
    """
    Additional recoverability statistics computed from the same Monte
    Carlo samples the RecoverabilityAnalyzer already produces -- no new
    sampling, just further summarising data that already exists:

        worst_disturbance: the non-recoverable sample CLOSEST to the
            equilibrium (smallest ||sample - x_star||) -- the most
            concerning failure, since it represents the smallest
            disturbance found to be non-recoverable in this sample set,
            i.e. an empirical lower bound on how close the true
            recoverability boundary comes to x_star. None if every
            sampled disturbance recovered.
        confidence_interval: a 95% Wilson score interval for the true
            recoverability probability, given n_recoverable successes
            out of n_samples Bernoulli trials -- standard statistics
            (Wilson 1927), not a new estimator. Reported because a
            recoverability index of, say, 0.90 means something very
            different at n_samples=20 (CI roughly [0.70, 0.97]) than at
            n_samples=500 (CI roughly [0.87, 0.93]).
    """
    samples = report.samples
    labels = np.asarray(report.labels, dtype=bool)
    failed = samples[~labels] if len(samples) else samples

    worst = None
    if len(failed) > 0:
        dists = np.linalg.norm(failed - x_star[None, :], axis=1)
        worst = failed[int(np.argmin(dists))].tolist()

    n = int(report.n_samples)
    k = int(report.n_recoverable)
    z = 1.96
    if n > 0:
        p_hat = k / n
        denom = 1 + z ** 2 / n
        center = (p_hat + z ** 2 / (2 * n)) / denom
        half_width = (z * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2))) / denom
        ci = [max(0.0, float(center - half_width)), min(1.0, float(center + half_width))]
    else:
        ci = None

    return {"worst_disturbance": worst, "confidence_interval": ci}


def _trace_recoverability_boundary(system, x_star: np.ndarray, u: np.ndarray, sim, manifold, recovery_tol: float,
                                    horizon: float, n_directions: int = 20, r_max: float = None,
                                    n_bisections: int = 6) -> dict:
    """
    Traces the recoverability boundary as a set of points, one per
    sampled direction, via direct bisection search along each ray from
    x_star -- not a Monte-Carlo estimate of an index, but an actual
    approximation of the geometric boundary itself (the review's
    "Critical Boundary" / "Recoverability Region" request).

    Restricted to 2-state systems: direction is parameterised by a
    single angle theta in [0, 2*pi) in the state plane, which has no
    natural generalisation to 3+ states without an additional choice
    (a spherical parameterisation, a specific 2D slice, etc.) that this
    version does not make -- returns None for higher-dimensional systems
    rather than silently picking an arbitrary slice and presenting it as
    "the" boundary.

    For each direction, a classic bisection search (validated below to
    correctly bracket a known analytic threshold before being trusted on
    real systems) finds the radius at which the sampled trajectory's
    recoverability classification (architecture document Definition 1.1,
    realised per Part II Section 2.4) flips.
    """
    n = system.n_states
    if n != 2:
        return None

    def is_recoverable(x0):
        traj = sim.simulate(x0, (0.0, horizon), u=u, n_eval=150)
        if not traj.success:
            return False
        admissible_path = all(system.admissible(traj.x[:, k]) for k in range(0, traj.x.shape[1], 5))
        r_final = manifold.residual(traj.final_state)
        return bool(admissible_path and r_final < recovery_tol and np.all(np.isfinite(traj.final_state)))

    if r_max is None:
        r_max = 3.0 * max(1.0, float(np.linalg.norm(x_star)))

    boundary_points = []
    for i in range(n_directions):
        theta = 2 * np.pi * i / n_directions
        direction = np.array([np.cos(theta), np.sin(theta)])
        lo, hi = 1e-3, r_max
        lo_ok = is_recoverable(x_star + lo * direction)
        hi_ok = is_recoverable(x_star + hi * direction)

        if lo_ok and not hi_ok:
            for _ in range(n_bisections):
                mid = 0.5 * (lo + hi)
                if is_recoverable(x_star + mid * direction):
                    lo = mid
                else:
                    hi = mid
            r_boundary = 0.5 * (lo + hi)
            status = "found"
        elif not lo_ok:
            r_boundary = lo
            status = "unrecoverable_at_min_radius"
        else:  # hi_ok: still recoverable at r_max, boundary not reached
            r_boundary = hi
            status = "beyond_search_radius"

        boundary_points.append({
            "theta": float(theta), "radius": float(r_boundary),
            "point": (x_star + r_boundary * direction).tolist(), "status": status,
        })

    found = [p["radius"] for p in boundary_points if p["status"] == "found"]
    return {
        "n_directions": n_directions,
        "points": boundary_points,
        "min_radius": float(min(p["radius"] for p in boundary_points)),
        "max_radius": float(max(p["radius"] for p in boundary_points)),
        "mean_radius_found": float(np.mean(found)) if found else None,
        "n_found": len(found),
    }


def _recoverability_interpretation(report) -> str:
    """
    Plain-language summary of the Monte-Carlo recoverability report,
    built directly from the same fields already reported numerically --
    no new computation, just clearer phrasing of what the index, sample
    counts, and risk level actually mean for the operating point.
    """
    n, k = int(report.n_samples), int(report.n_recoverable)
    if k == n:
        return (f"The operating point is fully recoverable within the tested disturbance radius. "
                f"No unrecoverable trajectories were observed across all {n} sampled disturbances.")
    if k == 0:
        return (f"None of the {n} sampled disturbances recovered within the tested radius. "
                f"This operating point is not robust to the disturbance magnitude tested here.")
    pct = 100.0 * k / n
    return (f"{k} of {n} sampled disturbances ({pct:.0f}%) recovered within the tested radius; "
            f"the remainder did not, giving a {report.risk_level} risk classification at this operating point.")


class _PIAwareSystemAdapter:
    """
    A thin wrapper around a DynamicalSystem that overrides
    find_equilibrium to try the validated PI-regulated nested solve
    first (see _try_pi_regulated_equilibrium), falling back to the
    system's own plain find_equilibrium if the network doesn't match
    that specific pattern or the nested solve doesn't converge.

    Used ONLY for the manifold-continuation and Monte-Carlo
    recoverability paths of custom_network category (which call
    find_equilibrium internally, once per sweep point / disturbance
    sample) -- deliberately implemented as a wrapper rather than a
    change to core.system.DynamicalSystem or ims.manifold.
    IntrinsicManifold itself, so the existing, validated behaviour of
    every other project category is completely unaffected (confirmed
    by every existing test still passing unchanged).

    Everything else (state_names, n_states, dynamics, jacobian, params)
    is forwarded directly to the wrapped system via __getattr__, so
    this is transparent to any caller that doesn't specifically need
    the overridden find_equilibrium.
    """

    def __init__(self, system, network: Network):
        self._system = system
        self._network = network

    def __getattr__(self, name):
        return getattr(self._system, name)

    def find_equilibrium(self, x0: np.ndarray, u=None, **kwargs):
        if u is not None:
            pi_result = _try_pi_regulated_equilibrium(self._system, self._network, np.asarray(x0, dtype=float), np.asarray(u, dtype=float))
            if pi_result is not None:
                J = self._system.jacobian(pi_result, u)
                eigs = np.linalg.eigvals(J)
                try:
                    cond_val = np.linalg.cond(J)
                    cond = float(cond_val) if np.isfinite(cond_val) else None
                except np.linalg.LinAlgError:
                    cond = None
                residual_norm = float(np.linalg.norm(self._system.dynamics(0.0, pi_result, u, self._system.params)))
                return EquilibriumResult(
                    x_star=pi_result, converged=True, residual_norm=residual_norm, eigenvalues=eigs,
                    n_function_evals=None, jacobian=J, jacobian_condition_number=cond, computation_time_s=0.0,
                )
        return self._system.find_equilibrium(x0, u=u, **kwargs)


def stage_ims_analysis(project: dict, payload: dict) -> dict:
    system, param_name, extra = _build_system(project, payload)
    x_star = np.array(payload["x_star"], dtype=float)

    if project["category"] == "converter_topology":
        v_guess = extra["v_guess"]
        x0_guess = np.array([v_guess, v_guess / 10.0])
    else:
        x0_guess = np.array(payload.get("state_guess", x_star.tolist()))

    sweep_range = payload.get("sweep_range", project["sweep_range"])
    sweep_points = int(payload.get("sweep_points", project["sweep_points"]))
    manifold_system = _PIAwareSystemAdapter(system, system.network) if (project["category"] == "custom_network" and hasattr(system, "network")) else system
    resolved_projection_method = _resolve_projection_method(payload.get("residual_projection_method", "auto"))
    M = IntrinsicManifold(manifold_system, param_name=param_name, projection_method=resolved_projection_method).build(
        alpha_range=tuple(sweep_range), n_points=sweep_points, x0_guess=x0_guess, keep_unstable=True,
    )
    if len(M.points) == 0:
        raise ValueError("manifold continuation produced zero points")

    result = {
        "residual_projection_method": resolved_projection_method,
        "residual_projection_method_requested": payload.get("residual_projection_method", "auto"),
        "manifold": {
            "points": [p.x_star.tolist() for p in M.points],
            "stable": [bool(p.is_stable) for p in M.points],
            "alpha": [float(p.param_value) for p in M.points],
        },
        "manifold_stats": _manifold_statistics(M, param_name, x_star),
    }

    # Manifold residual over time along the same disturbance trajectory
    # used at the Run Simulation stage -- "distance to manifold" over
    # the course of recovery, the network_mrc category already showed
    # (via its own hand-derived em(x)) and single_model/converter_topology
    # can now show too, via the generic IntrinsicManifold.residual_trajectory.
    if payload.get("disturbance") is not None:
        u = np.array(payload["nominal_input"], dtype=float)
        dist = payload["disturbance"]
        disturbed = x_star + np.array(dist["value"], dtype=float) if dist["type"] == "offset" else np.array(dist["value"], dtype=float)
        horizon = float(payload.get("horizon", project.get("default_horizon", 5.0)))
        sim_r = Simulator(system, method="RK45")
        traj_r = sim_r.simulate(disturbed, (0.0, horizon), u=u, n_eval=300)
        residual_r = M.residual_trajectory(traj_r.x)
        result["manifold_residual_trajectory"] = {
            "t": traj_r.t.tolist(),
            "residual": residual_r.tolist(),
        }
        result["residual_statistics"] = {
            "max": float(np.max(residual_r)),
            "rms": float(np.sqrt(np.mean(residual_r ** 2))),
            "mean": float(np.mean(residual_r)),
            "final": float(residual_r[-1]),
            "empirical_contraction_rate": _estimate_contraction_rate(traj_r.t, residual_r),
        }
        # Deterministic recoverability, per Definition IV.2 of the IMS
        # papers: computed directly from THIS trajectory's own manifold
        # residual, not a statistical estimate over many random
        # samples. This is the PRIMARY recoverability signal for this
        # framework -- Monte Carlo (below, "monte_carlo_validation")
        # exists to statistically validate this deterministic result
        # over a neighborhood of disturbances, not to replace it.
        J_star = system.jacobian(x_star, u)
        eigs_star = np.linalg.eigvals(J_star)
        det_assessment = _deterministic_recoverability_assessment(traj_r.t, residual_r)
        result["recoverability_deterministic"] = {
            **det_assessment,
            "ims_conditions": _check_ims_conditions(
                eigs_star, det_assessment["contraction_rate"], det_assessment["final_residual"], det_assessment["tolerance_used"],
            ),
            "geometric_narrative": _geometric_recoverability_narrative(det_assessment),
            **_identify_limiting_state(M, traj_r.x[:, -1], list(system.state_names)),
        }
        result["manifold_residual_explanation"] = _manifold_residual_explanation()
        if resolved_projection_method in ("polyline", "newton_refined"):
            result["projection_method_validation_summary"] = _projection_method_validation_summary()
        result["recoverability_region_summary"] = _recoverability_region_summary(det_assessment, False)

    if payload.get("recoverability_enabled", True):
        u = np.array(payload["nominal_input"], dtype=float)
        radius = float(payload.get("radius", project.get("default_radius", 1.0)))
        n_samples = int(payload.get("n_samples", project.get("default_samples", 60)))
        tol = float(payload.get("recovery_tol", project.get("default_tol", 0.1)))
        horizon = float(payload.get("horizon", project.get("default_horizon", 5.0)))
        sim = Simulator(system, method="RK45")
        analyzer = RecoverabilityAnalyzer(system, M, sim, recovery_tol=tol)
        report = analyzer.assess(x_star, radius=radius, n_samples=n_samples, t_horizon=horizon, u=u)
        result["recoverability"] = {
            "samples": report.samples.tolist(), "labels": [bool(v) for v in report.labels],
            "index": float(report.recoverability_index), "n_recoverable": int(report.n_recoverable),
            "n_samples": int(report.n_samples), "margin": float(report.margin_to_boundary), "risk": report.risk_level,
            "interpretation": _recoverability_interpretation(report),
            "role": "Statistical validation of the deterministic manifold-residual recoverability assessment (recoverability_deterministic) over a neighborhood of disturbances -- not the primary recoverability signal.",
            **_recoverability_extras(report, x_star),
        }

        if payload.get("trace_boundary") and system.n_states == 2:
            boundary = _trace_recoverability_boundary(
                system, x_star, u, sim, M, recovery_tol=tol, horizon=horizon,
                n_directions=int(payload.get("boundary_directions", 20)),
                r_max=float(payload["boundary_r_max"]) if payload.get("boundary_r_max") else None,
            )
            if boundary is not None:
                result["recoverability_boundary"] = boundary
                if "recoverability_region_summary" in result:
                    result["recoverability_region_summary"] = _recoverability_region_summary(det_assessment, True)

        # Controller effect on recoverability: re-assess with the SAME
        # sampled disturbances (via the same random seed the analyzer
        # would otherwise redraw) under closed-loop control, so the
        # comparison isolates what the controller changes rather than
        # differences between two independent Monte-Carlo draws.
        if payload.get("mrc_enabled") and project["category"] == "single_model":
            Q = np.diag(payload.get("Q_diag", project["q_default"]))
            R = np.diag(payload.get("R_diag", project["r_default"]))
            u_min = np.array(payload.get("u_min", project.get("u_min_default"))) if payload.get("u_min") or project.get("u_min_default") else None
            u_max = np.array(payload.get("u_max", project.get("u_max_default"))) if payload.get("u_max") or project.get("u_max_default") else None
            lqr = ScheduledLQRControl(system, M, Q=Q, R=R, u_min=u_min, u_max=u_max)
            lqr.precompute_gain_schedule(u_nominal=u)
            controller = lqr.as_controller(u_nominal=u)
            report_cl = analyzer.assess(x_star, radius=radius, n_samples=n_samples, t_horizon=horizon, controller=controller)
            result["recoverability_closed_loop"] = {
                "samples": report_cl.samples.tolist(), "labels": [bool(v) for v in report_cl.labels],
                "index": float(report_cl.recoverability_index), "n_recoverable": int(report_cl.n_recoverable),
                "n_samples": int(report_cl.n_samples), "margin": float(report_cl.margin_to_boundary), "risk": report_cl.risk_level,
                **_recoverability_extras(report_cl, x_star),
            }

    return result


# ----------------------------------------------------------------------
# network_mrc category: structurally different (genuine MRC synthesis),
# handled as its own staged sequence rather than forced through
# _build_system, since it composes SynthesizedMRCController rather than
# taking a duty/setpoint input.
# ----------------------------------------------------------------------

def _build_network_mrc_with(payload: dict, apply_mrc: bool = True):
    p = payload.get("params", {})
    p_values = dict(R=float(p.get("R", 0.20)), L=float(p.get("L", 1.5e-3)), C=float(p.get("C", 2.5e-3)), P=float(p.get("P", 10e3)))
    km_value = float(payload.get("km", 500.0))
    v_bus_init = float(payload.get("v_bus_init", 400.0))

    synthesis = MRCSynthesizer(ConverterCPLPaper()).synthesize()
    if apply_mrc:
        # The synthesized MRC becomes the converter's ACTIVE controller.
        controller = SynthesizedMRCController(synthesis, p_values, km_value, state_order=[0, "bus", 1])
    else:
        # Baseline: the SAME converter with NO MRC -- outer-loop actuation held
        # at u = v_o_dot = 0 (open loop), so Before/After compares like-for-like.
        controller = ConstantSetpointController(v_ref=0.0)
    electrical_model = ConverterCPLElectricalModel(R=p_values["R"], L=p_values["L"])

    net = Network("mrc_demo_network")
    net.add_bus(Bus(id="bus", C=p_values["C"], v_init=v_bus_init, v_min=v_bus_init * 0.1))
    net.add_component(Converter(id="conv", bus="bus", electrical_model=electrical_model, controller=controller))
    net.add_component(ConstantPowerLoad(id="cpl", bus="bus", P=p_values["P"], v_floor=1e-6))
    system = AutomaticModelBuilder.build(net)
    return system, net, synthesis, p_values, km_value, v_bus_init


def _build_network_mrc(payload: dict):
    system, _net, synthesis, p_values, km_value, v_bus_init = _build_network_mrc_with(payload, apply_mrc=True)
    return system, synthesis, p_values, km_value, v_bus_init


def network_mrc_build(payload: dict) -> dict:
    system, synthesis, p_values, km_value, v_bus_init = _build_network_mrc(payload)
    return {
        "state_names": list(system.state_names),
        "control_law_str": str(synthesis.control_expr),
        "control_law_latex": synthesis.as_latex(),
        "params_used": p_values,
        "km": km_value,
        "network_summary": _network_summary_from_network(system.network),
        "component_chain": _component_chain_from_network(system.network),
        "topology_tree": _topology_tree_from_network(system.network),
        "model_order": {"n_states": system.n_states, "n_inputs": 0, "n_outputs": system.n_states},
        "controller_description": _controller_description(PROJECTS["network_auto_mrc"]),
    }


def network_mrc_equilibrium(payload: dict) -> dict:
    system, synthesis, p_values, km_value, v_bus_init = _build_network_mrc(payload)
    x0_guess = np.array([v_bus_init, p_values["P"] / v_bus_init, v_bus_init + 5.0])
    eq = system.find_equilibrium(x0_guess, u=np.zeros(0))
    scale = max(1.0, float(np.linalg.norm(eq.x_star)))
    ok = eq.residual_norm / scale < 1e-6
    if not ok:
        raise ValueError("network equilibrium solve did not converge to adequate relative precision")
    return {
        "state_names": list(system.state_names), "x_star": eq.x_star.tolist(), "stable": bool(eq.is_stable),
        "solver_diagnostics": {
            "function_evaluations": eq.n_function_evals,
            "jacobian_condition_number": eq.jacobian_condition_number,
            "computation_time_s": eq.computation_time_s,
        },
        "component_operating_points": _component_operating_points(system.network, eq.x_star, list(system.state_names)),
        "note": (
            "This equilibrium's full closed-loop stability is a known, documented open item "
            "(see tests/test_mrc_synthesis.py): a structurally positive tangential eigenvalue in "
            "this specific reference model, not a defect of MRC synthesis itself. What IS "
            "guaranteed is the manifold residual's exponential contraction at rate k_m."
        ) if not eq.is_stable else "",
    }


def network_mrc_simulate(payload: dict) -> dict:
    system, synthesis, p_values, km_value, v_bus_init = _build_network_mrc(payload)
    x_star = np.array(payload["x_star"], dtype=float)
    disturbance = payload.get("disturbance", [-5.0, 0.0, 0.0])
    horizon = float(payload.get("horizon", 0.02))
    disturbed = x_star + np.array(disturbance, dtype=float)
    sim = Simulator(system, method="RK45")
    traj = sim.simulate(disturbed, (0.0, horizon), u=np.zeros(0), n_eval=400)

    R = p_values["R"]

    def residual(x):
        v_bus, i_l, v_o = x
        return v_bus - (v_o - R * i_l)

    residuals = [residual(traj.x[:, k]) for k in range(traj.x.shape[1])]

    kappa = synthesis.compile_numeric(p_values, km_value)
    control_signal = [float(kappa(traj.x[:, k], np.array([]))) for k in range(traj.x.shape[1])]

    metrics = _performance_metrics(np.array(traj.t), np.array(traj.x), x_star)
    u_arr = np.array(control_signal)
    control_effort = {"rms": float(np.sqrt(np.mean(u_arr ** 2))), "peak": float(np.max(np.abs(u_arr)))}

    return {
        "disturbed_state": disturbed.tolist(),
        "trajectory_open_loop": {"t": traj.t.tolist(), "x": traj.x.tolist(), "success": traj.success},
        "control_signal": control_signal,
        "control_effort": control_effort,
        "performance_metrics_open_loop": metrics,
        "residual": residuals, "km": km_value,
        "controller_info": {
            "type": "IMS-Native MRC (Symbolic Engine derived)",
            "law": str(synthesis.control_expr), "km": km_value,
        },
    }


def network_mrc_ims_analysis(payload: dict) -> dict:
    # No genuine multi-parameter manifold sweep is defined for this demo
    # (architecture doc Section 5.4 notes multi-parameter continuation as
    # future work); the manifold-residual decay already computed by
    # network_mrc_simulate IS this project's IMS analysis -- returned
    # here too so the Explorer's IMS Analysis stage has something to
    # show without silently reusing the simulate stage's data under a
    # different label.
    result = network_mrc_simulate(payload)
    residual_arr = np.array(result["residual"])
    t_arr = np.array(result["trajectory_open_loop"]["t"])
    det_assessment = _deterministic_recoverability_assessment(t_arr, residual_arr)
    system_mrc, _, _, _, _ = _build_network_mrc(payload)
    x_star_mrc = np.array(payload["x_star"], dtype=float)
    eigs_mrc = np.linalg.eigvals(system_mrc.jacobian(x_star_mrc, np.zeros(0)))
    return {
        "residual": result["residual"],
        "trajectory": result["trajectory_open_loop"],
        "residual_statistics": {
            "max": float(np.max(np.abs(residual_arr))), "rms": float(np.sqrt(np.mean(residual_arr ** 2))),
            "mean": float(np.mean(residual_arr)), "final": float(residual_arr[-1]),
            "empirical_contraction_rate": _estimate_contraction_rate(t_arr, np.abs(residual_arr)),
        },
        # Primary recoverability signal for this project, matching the
        # paper's own Definition IV.2 directly: this project's manifold
        # residual e_m = v_bus - (v_o - R*i_l) IS the paper's own
        # analytic formula (eq. 6), not a generic nearest-point
        # approximation -- so this classification is as close to the
        # paper's literal criterion as this platform currently gets.
        # Validated against the paper's own ablation contrast: km=500
        # (full MRC) classifies Recoverable here; km~0 (MRC-no-IMS,
        # mirroring the paper's Fig. 4) classifies Non-Recoverable.
        "recoverability_deterministic": {
            **det_assessment,
            "ims_conditions": _check_ims_conditions(eigs_mrc, det_assessment["contraction_rate"], det_assessment["final_residual"], det_assessment["tolerance_used"], transverse_target_rate=result.get("km")),
            "geometric_narrative": _geometric_recoverability_narrative(det_assessment),
        },
        "manifold_residual_explanation": _manifold_residual_explanation(),
        "recoverability_region_summary": _recoverability_region_summary(det_assessment, False),
        "note": "This project's manifold is 1-D and already shown by the simulate stage; no separate multi-point sweep is defined here (see architecture doc \u00a75.4).",
    }


def network_mrc_apply(payload: dict) -> dict:
    """
    Assign the synthesized MRC as the ACTIVE controller of the converter in
    the assembled Network Builder model. Returns the converter/controller
    configuration so the UI can show MRC as the converter's active synthesized
    controller (not merely an analysis panel).
    """
    system, net, synthesis, p_values, km_value, v_bus_init = _build_network_mrc_with(payload, apply_mrc=True)
    conv = next((c for c in net.components if isinstance(c, Converter)), None)
    return {
        "applied": True,
        "converter": {
            "id": conv.id if conv else "conv",
            "bus": conv.bus if conv else "bus",
            "electrical_model": type(conv.electrical_model).__name__ if conv else "ConverterCPLElectricalModel",
            "active_controller": {
                "type": "SynthesizedMRCController",
                "label": "IMS-Native MRC (Symbolic Engine derived)",
                "law": str(synthesis.control_expr),
                "law_latex": synthesis.as_latex(),
                "k_m": km_value,
                "manifold_residual": "e_m = v_b - (v_o - R*i_l)",
            },
        },
        "network_summary": _network_summary_from_network(net),
        "note": ("The synthesized MRC is now the converter's active controller in the assembled network. "
                 "Rerun the same network and disturbance to compare baseline vs MRC."),
    }


def network_mrc_closed_loop(payload: dict) -> dict:
    """
    Baseline (converter open-loop, u = v_o_dot = 0) vs MRC on the IDENTICAL
    assembled network, equilibrium and disturbance. Runs the real
    AutomaticModelBuilder-assembled system in both configurations.
    """
    x_star = np.array(payload["x_star"], dtype=float)
    disturbance = payload.get("disturbance", [-5.0, 0.0, 0.0])
    horizon = float(payload.get("horizon", 0.02))
    disturbed = x_star + np.array(disturbance, dtype=float)
    runs = {}
    for key, apply_mrc in (("baseline", False), ("mrc", True)):
        system, net, synthesis, p_values, km_value, _ = _build_network_mrc_with(payload, apply_mrc=apply_mrc)
        R = p_values["R"]
        sim = Simulator(system, method="RK45")
        traj = sim.simulate(disturbed, (0.0, horizon), u=np.zeros(0), n_eval=400)
        resid = [float(traj.x[0, k] - (traj.x[2, k] - R * traj.x[1, k])) for k in range(traj.x.shape[1])]
        runs[key] = {
            "t": traj.t.tolist(),
            "v_bus": traj.x[0].tolist(), "i_L": traj.x[1].tolist(), "v_o": traj.x[2].tolist(),
            "residual": resid, "success": bool(traj.success),
            "final_deviation": float(np.linalg.norm(traj.x[:, -1] - x_star)),
            "final_residual": float(resid[-1]), "max_abs_residual": float(np.max(np.abs(resid))),
        }
    return {
        "state_names": ["v_bus", "i_L", "v_o"],
        "x_star": x_star.tolist(), "disturbed_state": disturbed.tolist(),
        "km": float(payload.get("km", 500.0)),
        "baseline": runs["baseline"], "mrc": runs["mrc"],
        "controller_applied": "SynthesizedMRCController on converter 'conv'",
        "disclaimer": ("Baseline (converter open-loop, u = v_o_dot = 0) vs MRC on the IDENTICAL assembled "
                       "network, equilibrium and disturbance. Numerical evidence on one scenario -- not a "
                       "certified regional IMS guarantee."),
    }


def network_mrc_inspect(payload: dict) -> dict:
    """
    Runs the REAL automatic MRC derivation pipeline
    (mrc_designer.auto_manifold.run_auto_mrc_pipeline) on the USER-BUILT
    (Network Builder) network: control-affine split f(x)+G(x)u, automatic
    construction of a candidate SIGNED controlled-target manifold phi(x)
    for any directly duty-modulated converter (Buck/Boost/Buck-Boost) in
    the network, Dphi*G feasibility, closed-form synthesis when feasible,
    independent numeric re-verification, and transverse/tangential
    stability separation.

    This is the actual mathematical derivation, not a hard-coded lookup --
    the result comes back as one of MRC_SYNTHESIS_SUPPORTED,
    MRC_FEASIBILITY_DIAGNOSTIC_ONLY, or MRC_SYNTHESIS_NOT_ESTABLISHED
    (with the exact failed mathematical condition), from run_auto_mrc_pipeline
    itself -- never forced, never a generic "no registered manifold" message.

    For a network with no directly duty-modulated converter at all (the
    only structural class this construction currently covers -- see the
    module docstring of auto_manifold.py), the pipeline still runs and
    reports NOT_ESTABLISHED with the specific missing structural
    condition; the f(x)/G(x) numeric control-affine split is still
    reported underneath, for diagnostic visibility, via a lightweight
    fallback finite-difference check on the assembled system.
    """
    spec = payload.get("network_spec")
    if not spec:
        raise ValueError("network_spec is required")
    net, input_id = _build_network_from_spec(spec)

    converter_id = payload.get("converter_id")
    v_nom = payload.get("v_nom")
    if v_nom is not None:
        v_nom = float(v_nom)
    mrc_params = payload.get("mrc_params")

    report = run_auto_mrc_pipeline_multi(net, converter_id=converter_id, v_nom=v_nom, mrc_params=mrc_params)

    # Diagnostic numeric f(x)/G(x) split on the same assembled network
    # (independent of whatever the pipeline internally used), so the
    # inspect response always shows the actual control-affine structure
    # of THIS model even when no duty-modulated converter is present.
    system = AutomaticModelBuilder.build(net, input_component_id=input_id, name=spec.get("name"))
    n = system.n_states
    m = len(system.input_names)
    if payload.get("x_star") is not None:
        x_star = np.array(payload["x_star"], dtype=float)
    else:
        x_star = system.find_equilibrium(system.initial_guess(), with_eigs=False).x_star
    u0 = np.array(payload.get("nominal_input", system.default_input()), dtype=float)
    if u0.shape[0] != m:
        u0 = np.zeros(m)
    p = system.params
    f0 = np.asarray(system.dynamics(0.0, x_star, u0, p), dtype=float)
    G = np.zeros((n, m)); h = 1e-6
    for j in range(m):
        uj = u0.astype(float).copy(); uj[j] += h
        G[:, j] = (np.asarray(system.dynamics(0.0, x_star, uj, p), dtype=float) - f0) / h
    g_norm = float(np.linalg.norm(G))

    # "established" must agree with EVERY status the pipeline itself
    # treats as a genuine synthesis outcome -- not just the first-
    # candidate (phi1) MRC_SYNTHESIS_SUPPORTED case. A network whose only
    # viable path is the bus-voltage-anchored second candidate (phi2,
    # "reshaped manifold") is a real, feasibility-and-stability-gated
    # SUPPORTED result (see run_auto_mrc_pipeline / attempt_bus_voltage_
    # manifold_reshaping) and must be reported as established too --
    # omitting it here previously made this endpoint (and the Network
    # Builder UI, which reads 'mrc_established' directly) disagree with
    # the pipeline's own 'status'/'classification' for the identical
    # report, a real UI/report inconsistency found during end-to-end
    # validation.
    established = report.get("status") in (
        "MRC_SYNTHESIS_SUPPORTED", "MRC_SYNTHESIS_SUPPORTED_VIA_RESHAPED_MANIFOLD",
    )
    return {
        "mrc_established": established,
        "establishment_basis": report.get("status"),
        "classification": report.get("classification"),
        "dim_x": int(n), "dim_u": int(m), "input_names": list(system.input_names),
        "state_names": list(system.state_names),
        "operating_point": [float(v) for v in x_star],
        "G_method": "finite_difference", "G_norm": g_norm,
        "control_authority": bool(g_norm > 1e-9),
        "pipeline_report": report,
        "reason": report.get("reason"),
        "note": ("Full automatic MRC derivation pipeline (Network Builder -> control-affine split -> "
                 "controlled-target manifold -> feasibility -> synthesis -> independent verification -> "
                 "transverse/tangential stability) executed on the actual assembled model; see "
                 "'pipeline_report' for every intermediate quantity (candidate phi, Dphi*G, rank, relative "
                 "degree, derived control law, contraction residual, closed-loop eigenvalues, simulation)."),
    }


def network_mrc_auto_apply(payload: dict) -> dict:
    """
    Attach the automatically-derived MRC controller to the target
    directly-duty-modulated converter of a USER-BUILT Network Builder
    model, making it the converter's actual active controller (task
    requirement 7) -- only when run_auto_mrc_pipeline has established
    synthesis for that specific model; raises otherwise (the same honest
    failure the /inspect endpoint would have reported).
    """
    spec = payload.get("network_spec")
    if not spec:
        raise ValueError("network_spec is required")
    net, _input_id = _build_network_from_spec(spec)
    converter_id = payload.get("converter_id")
    v_nom = payload.get("v_nom")
    if v_nom is not None:
        v_nom = float(v_nom)
    mrc_params = payload.get("mrc_params")

    report = run_auto_mrc_pipeline_multi(net, converter_id=converter_id, v_nom=v_nom, mrc_params=mrc_params)
    if report.get("status") == "MRC_MULTI_CANDIDATE_SELECTION_REQUIRED":
        raise ValueError(
            f"Multiple feasible converters -- explicit 'converter_id' selection required: {report.get('reason')}"
        )
    # "MRC_SYNTHESIS_SUPPORTED_VIA_RESHAPED_MANIFOLD" (the bus-voltage-
    # anchored second-candidate, M2, outcome) is a genuine, feasibility-
    # and-stability-gated SUPPORTED result and must be applyable here too
    # -- previously this endpoint refused it even though run_auto_mrc_
    # pipeline itself reports it as SUPPORTED, so a user could never
    # actually attach the reshaped manifold's controller through the
    # Network Builder workflow (found during end-to-end validation).
    if report.get("status") not in (
        "MRC_SYNTHESIS_SUPPORTED", "MRC_FEASIBILITY_DIAGNOSTIC_ONLY",
        "MRC_SYNTHESIS_SUPPORTED_VIA_RESHAPED_MANIFOLD",
    ):
        raise ValueError(f"MRC synthesis not established for this model -- {report.get('reason')}")

    resolved_converter_id = converter_id or report["converter_id"]
    # Same "no v_nom given -> use the converter's own bus v_init" convention
    # run_auto_mrc_pipeline itself uses (auto_manifold.py); resolved here too
    # so attach_auto_mrc/attach_bus_voltage_mrc (which require a float) are
    # never handed None.
    v_nom_val = v_nom if v_nom is not None else float(net.buses[find_duty_modulated_converter(net, resolved_converter_id).bus].v_init or 1.0)

    if report["status"] == "MRC_SYNTHESIS_SUPPORTED_VIA_RESHAPED_MANIFOLD":
        # Apply the SECOND candidate (phi2, bus-voltage-anchored) with the
        # exact (R_v, K_i, k_m) the pipeline itself found and verified --
        # never a re-search here, and never phi1 (whose reduced dynamics
        # this exact report already found UNSTABLE for this network).
        mr = report["manifold_reshaping"]
        m2_params = {"R_v": mr["chosen_R_v"], "K_i": mr["K_i"], "k_m": mr["k_m"]}
        new_net, mrc_converter = attach_bus_voltage_mrc(net, resolved_converter_id, v_nom_val, m2_params)
        active_controller = {
            "type": "BusVoltageAnchoredMRCController",
            "label": "Automatically-derived bus-voltage-anchored MRC (M2, reshaped manifold)",
            "law_latex": mr.get("control_law2_latex"),
            "topology": report.get("topology"),
            "candidate": "phi2 (bus-voltage-anchored, reshaped)",
            "R_v": mr["chosen_R_v"], "K_i": mr["K_i"], "k_m": mr["k_m"],
        }
    else:
        new_net, mrc_converter = attach_auto_mrc(net, resolved_converter_id, v_nom_val, mrc_params)
        active_controller = {
            "type": "AutoCurrentLoopMRCController",
            "label": "Automatically-derived current-loop MRC (auto_manifold)",
            "law_latex": report.get("derived_control_law"),
            "topology": report.get("topology"),
            "candidate": "phi1 (current-anchored)",
        }

    return {
        "applied": True,
        "status": report["status"],
        "classification": report.get("classification"),
        "converter": {
            "id": mrc_converter.id, "bus": mrc_converter.bus,
            "electrical_model": type(mrc_converter.electrical_model).__name__,
            "active_controller": active_controller,
        },
        "network_summary": _network_summary_from_network(new_net),
        "pipeline_report": report,
        "note": ("The automatically-derived MRC is now this converter's active controller in the assembled "
                 "network. Use /api/network_mrc/auto_closed_loop with the same network_spec to compare "
                 "baseline vs MRC."),
    }


def network_mrc_auto_closed_loop(payload: dict) -> dict:
    """
    Baseline (the converter's original controller) vs automatically-
    derived MRC, on the IDENTICAL user-built assembled network -- the
    same Build -> Solve Equilibrium -> Apply MRC -> Run Closed Loop ->
    Before/After workflow as the reference network_mrc project, but for
    an arbitrary Network Builder model via run_auto_mrc_pipeline. Reuses
    the pipeline's own closed-loop simulation result rather than
    re-simulating separately, since the pipeline already runs baseline
    equilibrium, MRC equilibrium, and a disturbed-recovery simulation of
    the actual closed-loop system.
    """
    spec = payload.get("network_spec")
    if not spec:
        raise ValueError("network_spec is required")
    net, _input_id = _build_network_from_spec(spec)
    converter_id = payload.get("converter_id")
    v_nom = payload.get("v_nom")
    if v_nom is not None:
        v_nom = float(v_nom)
    mrc_params = payload.get("mrc_params")

    report = run_auto_mrc_pipeline_multi(net, converter_id=converter_id, v_nom=v_nom, mrc_params=mrc_params)
    if report.get("status") == "MRC_MULTI_CANDIDATE_SELECTION_REQUIRED":
        raise ValueError(
            f"Multiple feasible converters -- explicit 'converter_id' selection required: {report.get('reason')}"
        )
    # Accept the M2 (reshaped-manifold) outcome here too -- see
    # network_mrc_auto_apply's identical note. For this outcome, phi1's
    # own contraction_residual/closed_loop_eigenvalues fields above refer
    # to the UNSTABLE, diagnostic-only first candidate, NOT the accepted
    # controller -- returning those unchanged for an M2 result would be
    # exactly the kind of "report disagrees with what was actually
    # applied" bug the end-to-end validation is meant to catch, so this
    # branches to a genuine three-way baseline/M1/M2 comparison instead.
    if report.get("status") not in (
        "MRC_SYNTHESIS_SUPPORTED", "MRC_FEASIBILITY_DIAGNOSTIC_ONLY",
        "MRC_SYNTHESIS_SUPPORTED_VIA_RESHAPED_MANIFOLD",
    ):
        raise ValueError(f"MRC synthesis not established for this model -- {report.get('reason')}")

    if report["status"] == "MRC_SYNTHESIS_SUPPORTED_VIA_RESHAPED_MANIFOLD":
        mr = report["manifold_reshaping"]
        resolved_converter_id = converter_id or report["converter_id"]
        v_nom_val = v_nom if v_nom is not None else float(
            net.buses[find_duty_modulated_converter(net, resolved_converter_id).bus].v_init or 1.0)
        cmp = compare_baseline_m1_m2(
            net, resolved_converter_id, v_nom_val, mrc_params_m1=mrc_params,
            R_v=mr["chosen_R_v"], K_i_m2=mr["K_i"], k_m_m2=mr["k_m"],
        )
        return {
            "status": report["status"],
            "classification": report.get("classification"),
            "converter_id": report.get("converter_id"),
            "topology": report.get("topology"),
            "active_candidate": report.get("active_candidate"),
            "chosen_R_v": report.get("chosen_R_v"),
            "manifold_reshaping": report.get("manifold_reshaping"),
            "claim_levels": report.get("claim_levels"),
            # Genuine three-way comparison (item 4/5 of the validation
            # spec): baseline (original controller), M1 (current-anchored,
            # transverse-attracting but reduced-dynamics UNSTABLE), M2
            # (bus-voltage-anchored reshaped manifold, both transverse and
            # reduced dynamics locally stable) -- identical network,
            # equilibrium basis, disturbance, solver, tolerances and
            # horizon for all three arms.
            "baseline_m1_m2_comparison": cmp,
            "closed_loop_comparison": {"baseline": cmp["baseline"], "mrc": cmp["m2"]},
            "disclaimer": ("M1 achieves exact transverse attraction to its manifold but has UNSTABLE reduced "
                           "dynamics; M2 reshapes the manifold so that both the transverse AND the reduced "
                           "dynamics are locally stable (Hurwitz, verified via full Routh-Hurwitz, not trace "
                           "alone). This is local-stability and finite-horizon simulation evidence on the "
                           "actual assembled model -- not a certified regional IMS guarantee."),
        }

    return {
        "status": report["status"],
        "classification": report.get("classification"),
        "converter_id": report.get("converter_id"),
        "topology": report.get("topology"),
        "equilibrium_preservation": report.get("equilibrium_preservation"),
        "contraction_residual": report.get("contraction_residual"),
        "closed_loop_eigenvalues": report.get("closed_loop_eigenvalues"),
        "transverse_eigenvalue": report.get("transverse_eigenvalue"),
        "tangential_eigenvalues": report.get("tangential_eigenvalues"),
        "tangential_locally_stable": report.get("tangential_locally_stable"),
        "simulation_result": report.get("simulation_result"),
        "closed_loop_comparison": report.get("closed_loop_comparison"),
        "claim_levels": report.get("claim_levels"),
        "derived_control_law": report.get("derived_control_law"),
        "saturation_note": report.get("saturation_note"),
        "disclaimer": ("Automatic per-model result from the same mathematical pipeline as /inspect and "
                       "/auto_apply, run on the actual assembled Network Builder model -- not a certified "
                       "regional IMS guarantee (see claim_levels for the four separately-tracked claims)."),
    }
