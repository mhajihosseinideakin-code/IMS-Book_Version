"""
network.controller
--------------------

`Controller`: the control-law half of the Converter object model
(architecture document Part IV Section 4.4). A Controller maps
measurements (the bus/terminal voltage, the paired ElectricalModel's own
states, its own internal states, and an optional exogenous input) to a
single scalar control signal, with no knowledge of the electrical
topology that signal will drive. This is what makes "swap the controller
without touching the converter" a real, testable operation rather than
an aspiration: any Controller can be paired with any ElectricalModel
whose `local_dynamics`/`port_current` accept a scalar control signal in
the units that Controller produces (by convention in this version, a
voltage reference).

Architectural note: this base class, and the two minimal examples below,
are the object model Phase V2.3 is about -- NOT the target controller
library. Droop, PI (as a first-class, documented library member rather
than the illustrative example below), MRC (via
`control.mrc_synthesis.MRCSynthesizer`), backstepping, and scheduled LQR
are Phase V2.5.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple


class Controller:
    """
    Base class for a control law. Subclasses declare `state_names`
    (local, unprefixed -- empty for a stateless/static controller) and
    `param_names` as class attributes, are constructed with keyword
    parameter values matching `param_names`, and implement
    `control_signal` and, if stateful, `local_dynamics`.
    """

    state_names: Tuple[str, ...] = ()
    param_names: Tuple[str, ...] = ()

    def __init__(self, **params):
        missing = set(self.param_names) - set(params)
        if missing:
            raise ValueError(f"{type(self).__name__}: missing required parameter(s) {sorted(missing)}")
        self.params: Dict[str, float] = {k: params[k] for k in self.param_names}

    def control_signal(
        self, v_bus: float, electrical_x: np.ndarray, controller_x: np.ndarray, u: Optional[float], p: Dict
    ) -> float:
        """The scalar control signal (e.g. a voltage reference) this controller produces right now."""
        raise NotImplementedError

    def local_dynamics(
        self, v_bus: float, electrical_x: np.ndarray, controller_x: np.ndarray, u: Optional[float], p: Dict
    ) -> np.ndarray:
        """d(controller_x)/dt. Default: stateless, empty array."""
        return np.zeros(0)

    def initial_state_guess(self) -> np.ndarray:
        """
        A reasonable initial guess for this controller's own local states,
        used when the assembler seeds an equilibrium-solve initial guess.
        Default: 1.0 per state, matching the assembler's prior blanket
        default for all component-local states. Subclasses with an
        integral/accumulator state (e.g. a PI controller's error
        integral) should override this -- an accumulator naturally
        starts at "no accumulated error yet" (0.0), and for a controller
        whose output feeds a hard-clamped physical quantity (e.g. a duty
        ratio clamped to [0.02, 0.98]), starting the integrator at an
        arbitrary nonzero value can push the very first Newton iterate
        into the clamped region, where the clamp's zero local derivative
        makes that state invisible to the Jacobian and the structural
        rank check (rank(J) = n_states) fails before the solver is even
        given a chance to iterate.
        """
        return np.ones(len(self.state_names))


class ConstantSetpointController(Controller):
    """The simplest possible controller: a fixed voltage reference, optionally overridden by u."""

    param_names = ("v_ref",)

    def control_signal(self, v_bus, electrical_x, controller_x, u, p) -> float:
        return p["v_ref"] if u is None else u


class ConstantDutyController(Controller):
    """
    The simplest possible controller for a `ConverterModel` topology
    (network.converter_topologies): a fixed switch duty ratio d,
    optionally overridden by u. Exists to exercise Buck/Boost/BuckBoost
    in isolation, in the same spirit as `ConstantSetpointController` for
    `FilteredVoltageSource` (Phase V2.3) -- not a production duty-cycle
    regulator. Closed-loop duty-ratio control (voltage-mode, current-
    mode, or IMS-native MRC via `control.mrc_synthesis`) is Phase V2.5.
    """

    param_names = ("d",)

    def control_signal(self, v_bus, electrical_x, controller_x, u, p) -> float:
        return p["d"] if u is None else u


class SynthesizedMRCController(Controller):
    """
    Adapter: wraps a control law derived by
    `control.mrc_synthesis.MRCSynthesizer` (the IMS Control Framework)
    as a `network.controller.Controller`, so a component built by the
    Symbolic Engine can drive a Converter's ElectricalModel through
    exactly the same interface as any hand-written example controller
    above. This is the piece that closes the gap between the two halves
    of the platform built so far: genuine IMS-native MRC (Phase V2.1)
    and the Network/Converter object model (Phase V2.2-V2.4) were, until
    this class, two capabilities that could not be composed.

    The wrapped ElectricalModel's local state, together with the bus
    voltage, must reassemble into exactly the state vector the
    synthesis's source model declared via `symbolic_symbols()` -- that
    mapping is the caller's responsibility, given as `state_order`, a
    sequence where each entry is either the string "bus" (use v_bus) or
    an integer (use electrical_x[i]), in the synthesis model's own state
    order. See `network.converter_cpl_electrical_model` for a complete,
    validated worked example.
    """

    def __init__(self, synthesis_result, p_values: Dict, km_value: float, state_order):
        # Deliberately does not call Controller.__init__ / use the
        # param_names mechanism: this controller's "parameters" are the
        # already-resolved p_values/km_value baked into the compiled
        # numeric law below, not a fresh set of tunable knobs.
        self.state_names = ()
        self.param_names = ()
        self.params = {}
        self._state_order = tuple(state_order)
        self._kappa = synthesis_result.compile_numeric(p_values, km_value)
        self.synthesis_result = synthesis_result
        self.km_value = km_value

    def _assemble_full_state(self, v_bus: float, electrical_x: np.ndarray) -> np.ndarray:
        vals = []
        for spec in self._state_order:
            vals.append(v_bus if spec == "bus" else electrical_x[spec])
        return np.array(vals)

    def control_signal(self, v_bus, electrical_x, controller_x, u, p) -> float:
        full_x = self._assemble_full_state(v_bus, electrical_x)
        return self._kappa(full_x, np.array([]))


class SimplePIVoltageController(Controller):
    """
    A minimal proportional-integral bus-voltage regulator, included to
    demonstrate a *stateful* controller (its own integrator state)
    composing correctly with a stateful ElectricalModel:

        v_ref = v_nom + Kp*(v_nom - v_bus) + Ki*e_int
        d(e_int)/dt = v_nom - v_bus

    This outputs a voltage reference (control_signal = v_ref), matching
    electrical models whose control_signal IS a voltage command (e.g.
    FilteredVoltageSource's v_ref). It is NOT a duty-cycle controller --
    using it with a duty-cycle-based converter model (BuckModel,
    BoostModel, BuckBoostModel, whose control_signal is clamped to
    [0.02, 0.98] as a duty ratio) would always saturate at maximum duty,
    since v_nom is a voltage (e.g. ~24), not a duty ratio. For those
    topologies, use PIDutyController instead.

    This is a minimal illustrative example, not the PI library member
    Phase V2.5 will provide (which should support anti-windup, output
    saturation, and a documented tuning interface).
    """

    state_names = ("e_int",)
    param_names = ("v_nom", "Kp", "Ki")

    def control_signal(self, v_bus, electrical_x, controller_x, u, p) -> float:
        v_nom = p["v_nom"] if u is None else u
        e_int = controller_x[0]
        return v_nom + p["Kp"] * (v_nom - v_bus) + p["Ki"] * e_int

    def local_dynamics(self, v_bus, electrical_x, controller_x, u, p) -> np.ndarray:
        v_nom = p["v_nom"] if u is None else u
        return np.array([v_nom - v_bus])

    def initial_state_guess(self) -> np.ndarray:
        return np.array([0.0])


class PIDutyController(Controller):
    """
    A proportional-integral bus-voltage regulator for duty-cycle-based
    converter topologies (BuckModel, BoostModel, BuckBoostModel), whose
    control_signal is clamped as a duty ratio in [0.02, 0.98].

    Outputs a duty-cycle correction around a nominal duty d_nominal,
    with the voltage error normalized by v_nom so Kp/Ki are
    dimensionless gains independent of the absolute voltage scale:

        d = d_nominal + Kp*(v_nom - v_bus)/v_nom + Ki*e_int
        d(e_int)/dt = (v_nom - v_bus)/v_nom

    Added because SimplePIVoltageController's control_signal is a
    voltage reference (v_nom + corrections, e.g. ~24), which always
    saturates a duty-cycle clamp at maximum duty regardless of the
    actual voltage error -- so a PI controller built from that class
    for Buck/Boost/BuckBoost topologies would never actually regulate
    voltage; it would behave identically to a constant-duty controller
    pinned at maximum duty. This is a minimal illustrative example, not
    a production-grade PI regulator (no anti-windup or documented
    tuning interface).
    """

    state_names = ("e_int",)
    param_names = ("v_nom", "Kp", "Ki", "d_nominal")

    def control_signal(self, v_bus, electrical_x, controller_x, u, p) -> float:
        v_nom = p["v_nom"] if u is None else u
        e_int = controller_x[0]
        d_nom = p.get("d_nominal", 0.5)
        v_nom_safe = v_nom if abs(v_nom) > 1e-9 else 1e-9
        return d_nom + p["Kp"] * (v_nom - v_bus) / v_nom_safe + p["Ki"] * e_int

    def local_dynamics(self, v_bus, electrical_x, controller_x, u, p) -> np.ndarray:
        v_nom = p["v_nom"] if u is None else u
        v_nom_safe = v_nom if abs(v_nom) > 1e-9 else 1e-9
        return np.array([(v_nom - v_bus) / v_nom_safe])

    def initial_state_guess(self) -> np.ndarray:
        return np.array([0.0])


class DroopController(Controller):
    """
    A DC voltage-droop regulator for duty-cycle-based converter
    topologies (BuckModel, BoostModel, BuckBoostModel).

    Standard DC droop: the effective voltage setpoint decreases
    proportionally with the converter's own output current, which is
    the standard mechanism for proportional load sharing among parallel
    DC sources without requiring any communication between them --

        v_ref = v_nom - R_droop * i_L
        d = d_nominal + Kp*(v_ref - v_bus)/v_nom

    where i_L is the converter's own inductor current (its electrical
    model's local state) and R_droop is the droop resistance (V per A).
    A larger R_droop shares load more aggressively between converters at
    the cost of larger steady-state voltage deviation under load; a
    smaller R_droop holds voltage tighter but shares load less evenly
    when multiple droop-controlled converters feed the same bus.

    Unlike PIDutyController, this is a purely proportional, stateless
    law -- no integrator, so no equivalent of the "integral state starts
    saturated" issue a PI controller can hit at a poor initial guess.
    It also does not drive steady-state voltage error to exactly zero
    (a property of any pure-proportional droop law); PIDutyController
    remains the choice when exact voltage regulation matters more than
    load sharing.

    This is a minimal illustrative example (single-converter droop, no
    communication with other droop-controlled units, no measurement
    filtering), not a production-grade implementation.
    """

    state_names = ()
    param_names = ("v_nom", "Kp", "R_droop", "d_nominal")

    def control_signal(self, v_bus, electrical_x, controller_x, u, p) -> float:
        v_nom = p["v_nom"] if u is None else u
        i_L = electrical_x[0] if len(electrical_x) > 0 else 0.0
        v_nom_safe = v_nom if abs(v_nom) > 1e-9 else 1e-9
        v_ref = v_nom - p["R_droop"] * i_L
        d_nom = p.get("d_nominal", 0.5)
        return d_nom + p["Kp"] * (v_ref - v_bus) / v_nom_safe


#: Numeric topology codes for AutoCurrentLoopMRCController.params["topology_code"]
#: (kept numeric, not a string, because every other entry of a component's
#: params dict is a plain float elsewhere in the platform -- e.g.
#: network.assembler.AssembledNetworkSystem._collect_params/prefixed_params
#: and every JSON-serialization path over params/state -- so a stray string
#: value here would be a silent landmine for any code that assumes
#: float(v) works on every params entry).
TOPOLOGY_CODE = {"buck": 0.0, "boost": 1.0, "buckboost": 2.0}
TOPOLOGY_CODE_INV = {v: k for k, v in TOPOLOGY_CODE.items()}


def current_loop_mrc_duty(topology: str, i_L: float, v_bus: float, sigma: float, p: Dict) -> float:
    """
    Closed-form duty command for the automatically-derived, controller-
    shaped current-loop MRC target manifold

        phi(x) = i_L - i_bias - K_i * sigma,     sigma_dot = v_nom - v_bus

    applied to a directly duty-modulated converter (network.converter_
    topologies.BuckModel / BoostModel / BuckBoostModel), where the duty
    ratio d enters di_L/dt directly (relative degree one on i_L itself,
    by the converter's own averaged KVL relation) -- structurally the
    same "controller-shaped target manifold" category as the book's
    stabilizing-MRC construction (an integral-augmented linear target),
    just anchored on i_L rather than on a free outer-loop voltage state,
    because these topologies have no such free state for the duty ratio
    to actuate through.

    This closed form is produced by solving

        L * di_L/dt(i_L, v_bus, d) = L * [ -k_m*(i_L - i_bias - K_i*sigma)
                                            + K_i*(v_nom - v_bus) ]

    for d, using each topology's own averaged inductor-voltage relation
    (exactly the equations in network.converter_topologies). This
    duplication is deliberate and load-bearing, not accidental: it lets
    `mrc_designer.auto_manifold`'s independent SymPy derivation of the
    same closed form be checked against this runtime formula (and both,
    in turn, against a finite-difference derivative of the real,
    assembled ConverterModel.local_dynamics) as three separate
    computational paths for the same quantity --
    tests/test_auto_manifold_mrc.py::test_symbolic_matches_numeric_law
    is the regression that keeps them from silently drifting apart.

    Raises ValueError for any topology this construction does not (yet)
    cover -- never silently falls back to an unrelated formula.
    """
    v_nom = float(p["v_nom"]); K_i = float(p["K_i"]); k_m = float(p["k_m"])
    i_bias = float(p["i_bias"])
    v_in = float(p["v_in"]); L = float(p["L"]); R_L = float(p["R_L"])
    topo = TOPOLOGY_CODE_INV.get(float(p["topology_code"]))

    rhs_target = L * (-k_m * (i_L - i_bias - K_i * sigma) + K_i * (v_nom - v_bus))
    if topo == "buck":
        if abs(v_in) < 1e-12:
            raise ValueError("current-loop MRC duty law is singular for buck topology: v_in == 0")
        return (rhs_target + v_bus + R_L * i_L) / v_in
    if topo == "boost":
        if abs(v_bus) < 1e-9:
            raise ValueError("current-loop MRC duty law is singular for boost topology: v_bus == 0")
        return 1.0 - (v_in - R_L * i_L - rhs_target) / v_bus
    if topo == "buckboost":
        denom = v_in + v_bus
        if abs(denom) < 1e-9:
            raise ValueError("current-loop MRC duty law is singular for buck-boost topology: v_in + v_bus == 0")
        return (rhs_target + v_bus + R_L * i_L) / denom
    raise ValueError(f"current_loop_mrc_duty: unsupported topology code {p.get('topology_code')!r}")


class AutoCurrentLoopMRCController(Controller):
    """
    Runtime `Controller` adapter for the automatically-derived current-
    loop MRC law (`current_loop_mrc_duty` above). Structurally parallel
    to `SynthesizedMRCController`'s role for the reduced-order voltage-
    source model: this class contains no derivation of its own -- it is
    only the object that lets `mrc_designer.auto_manifold`'s derived law
    become the Converter's ACTUAL active controller (state, dynamics,
    and control signal), attached through the same `Converter`/
    `AutomaticModelBuilder` machinery as every other controller in this
    file, on the real assembled Network Builder model -- not a
    parallel/simulated stand-in.

        d(sigma)/dt = v_nom - v_bus                    (voltage-error integrator)
        d = current_loop_mrc_duty(topology, i_L, v_bus, sigma, p)   (duty command)
    """

    state_names = ("sigma",)
    param_names = ("v_nom", "K_i", "k_m", "i_bias", "topology_code", "v_in", "L", "R_L")

    def control_signal(self, v_bus, electrical_x, controller_x, u, p) -> float:
        i_L = float(electrical_x[0])
        sigma = float(controller_x[0])
        return current_loop_mrc_duty("", i_L, v_bus, sigma, p)

    def local_dynamics(self, v_bus, electrical_x, controller_x, u, p) -> np.ndarray:
        v_nom = p["v_nom"] if u is None else u
        return np.array([v_nom - v_bus])

    def initial_state_guess(self) -> np.ndarray:
        return np.array([0.0])
