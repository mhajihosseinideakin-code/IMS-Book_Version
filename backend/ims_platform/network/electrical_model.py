"""
network.electrical_model
--------------------------

`ElectricalModel`: the physical-topology half of the Converter object
model (architecture document Part IV Section 4.4). An ElectricalModel
describes a converter's own internal dynamics and its electrical port
behaviour -- how it responds, physically, to a control signal -- with no
knowledge of *how* that control signal is produced. The same
ElectricalModel can therefore be paired with any `Controller`
(network.controller) that produces a compatible control signal, which is
the whole point of keeping the two separable: a new control strategy for
an existing converter topology is a new Controller, not a new
ElectricalModel, and vice versa.

Port convention
----------------
Every ElectricalModel in this (DC-only, for now -- see architecture
document Section 5 for the AC/dq-frame generalisation, Phase V2.6)
version exposes a single DC port: a terminal voltage (the bus it is
attached to) and a terminal current (`port_current`, positive = current
flowing from the converter into the bus).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple


class ElectricalModel:
    """
    Base class for a converter's physical topology.

    Subclasses declare `state_names` and `param_names` as class
    attributes, and are constructed with keyword parameter values
    matching `param_names` (stored in `self.params`). They implement
    `port_current` and `local_dynamics`. `port_type` and `control_mode`
    are metadata (architecture document Section 4.2) used by the
    Automatic Model Builder to validate that a component is compatible
    with the bus/network it is bound to.
    """

    state_names: Tuple[str, ...] = ()
    param_names: Tuple[str, ...] = ()
    port_type: str = "DC"
    #: "grid_forming" (sets its own port voltage), "grid_following"
    #: (tracks a reference derived from measurements), or "passive"
    #: (no active control -- a line, a load).
    control_mode: str = "grid_forming"

    def __init__(self, **params):
        missing = set(self.param_names) - set(params)
        if missing:
            raise ValueError(f"{type(self).__name__}: missing required parameter(s) {sorted(missing)}")
        self.params: Dict[str, float] = {k: params[k] for k in self.param_names}

    def port_current(self, v_bus: float, local_x: np.ndarray, control_signal: float, p: Dict) -> float:
        """
        Current injected into the bus (A), as a function of the bus
        (terminal) voltage, this model's own local states, the control
        signal supplied by a paired Controller, and its parameters.
        """
        raise NotImplementedError

    def local_dynamics(self, v_bus: float, local_x: np.ndarray, control_signal: float, p: Dict) -> np.ndarray:
        """d(local_x)/dt for this electrical model's own states."""
        raise NotImplementedError

    def initial_state_guess(self) -> np.ndarray:
        """A reasonable initial guess for this electrical model's own local states. Default: 1.0 per state."""
        return np.ones(len(self.state_names))


class FilteredVoltageSource(ElectricalModel):
    """
    A minimal, generic example ElectricalModel: a first-order-filtered
    controllable voltage source behind a series output resistance.

        tau * d(v_conv)/dt = -v_conv + v_ref     (v_ref = control_signal)
        port_current        = (v_conv - v_bus) / R_out

    This represents, at the level of averaged/reduced-order modelling
    used throughout this platform, the output-filter dynamics common to
    a wide range of grid-forming converter topologies, without
    committing to any one specific power stage (buck, boost, etc. --
    Phase V2.4). It exists to demonstrate and validate the
    ElectricalModel/Controller separation (Phase V2.3); specific
    topologies are the natural next layer on top of it.
    """

    state_names = ("v_conv",)
    param_names = ("tau", "R_out")
    control_mode = "grid_forming"

    def port_current(self, v_bus: float, local_x: np.ndarray, control_signal: float, p: Dict) -> float:
        v_conv = local_x[0]
        return (v_conv - v_bus) / p["R_out"]

    def local_dynamics(self, v_bus: float, local_x: np.ndarray, control_signal: float, p: Dict) -> np.ndarray:
        v_conv = local_x[0]
        v_ref = control_signal
        return np.array([(-v_conv + v_ref) / p["tau"]])


class ConverterModel(ElectricalModel):
    """
    Base class for averaged (continuous-conduction-mode) switching DC-DC
    power-converter models: `BuckModel`, `BoostModel`, `BuckBoostModel`
    (network.converter_topologies), and later additions to the same
    hierarchy such as VSC, MMC, and battery models.

    Scope note: this deliberately fits the existing single-bus
    ElectricalModel/Converter contract (one port_current/local_dynamics
    pair, one attached bus) rather than modelling the converter's input
    side as a second, genuinely independent network bus. The input side
    is instead represented by the fixed parameter `v_in` (an idealised,
    stiff input rail) -- physically reasonable for the common case of a
    point-of-load converter fed from a well-regulated upstream bus, and
    it lets every topology in this hierarchy attach to the network
    exactly like `FilteredVoltageSource` already does, with no change to
    `network.converter.Converter` or `network.assembler`. A genuine
    two-port converter bridging two independently dynamic buses is a
    natural future extension, but requires generalising the Assembler to
    let a component span two buses (today only `network.branch.Line`
    does); that is a deliberately deferred, separately-scoped piece of
    work, not bundled into this one.

    By convention, the control signal for every ConverterModel subclass
    is the switch duty ratio d, and every subclass declares an inductor
    current as (part of) its local state -- both matching standard
    averaged-model practice for CCM DC-DC converters.
    """

    control_mode: str = "grid_forming"

    @staticmethod
    def clamp_duty(d: float, d_min: float = 0.02, d_max: float = 0.98) -> float:
        """Clamp a duty ratio away from the physically-degenerate 0/1 limits."""
        return min(max(d, d_min), d_max)
