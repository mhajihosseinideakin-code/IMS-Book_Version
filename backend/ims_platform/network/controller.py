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
