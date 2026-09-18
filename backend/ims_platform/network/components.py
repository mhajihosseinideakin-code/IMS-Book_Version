"""
network.components
-------------------

Bus-attached components (architecture document Part IV Section 4.2/4.4).

`BusComponent` is the base of the component hierarchy. Two kinds of
component exist:

    - Stateless (algebraic) components -- ConstantPowerLoad,
      ConstantImpedanceLoad, ConstantCurrentLoad, IdealSource -- whose
      current injection is a pure function of the bus voltage and an
      optional exogenous input. n_local_states = 0 for these.

    - Stateful components -- network.converter.Converter, composed from
      an ElectricalModel and a Controller (network.electrical_model,
      network.controller) -- which carry their own local dynamic states
      (e.g. an internal filter state, a controller integrator) alongside
      their current injection. n_local_states > 0 for these; the
      Automatic Model Builder (network.assembler) allocates a slice of
      the global state vector for each one automatically.

Every component, stateful or not, presents the same two-method contract
(`current_injection`, `local_dynamics`) so the Assembler never needs to
know which kind it is dealing with -- this is what "Ports" means in the
architecture document's Component hierarchy (Part IV Section 4.4): a
uniform electrical interface, regardless of what is behind it.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Sequence, Tuple


class BusComponent:
    """
    Base class for a bus-attached device.

    Subclasses implement `current_injection` (required) and, if they
    carry their own dynamic state, `n_local_states`/`local_state_names`/
    `local_dynamics` as well (all default to "no local state").
    """

    #: number of this component's own dynamic states (0 = purely algebraic)
    n_local_states: int = 0
    #: local (unprefixed) names for those states, e.g. ("v_filt", "e_int")
    _local_state_names: Tuple[str, ...] = ()

    def __init__(self, id: str, bus: str, **params):
        self.id = id
        self.bus = bus
        self.params: Dict[str, float] = dict(params)

    def current_injection(self, v_bus: float, local_x: np.ndarray, u: Optional[float] = None) -> float:
        """Current this component injects into its bus (A), positive = into the bus."""
        raise NotImplementedError

    def local_dynamics(self, v_bus: float, local_x: np.ndarray, u: Optional[float] = None) -> np.ndarray:
        """d(local_x)/dt for this component's own states. Default: no states, empty array."""
        return np.zeros(0)

    def local_state_names(self) -> Tuple[str, ...]:
        """Globally-unique (id-prefixed) names for this component's local states."""
        return tuple(f"{self.id}_{n}" for n in self._local_state_names)

    def prefixed_params(self) -> Dict[str, float]:
        """This component's own parameters, with globally-unique keys for the assembled system."""
        return {f"{self.id}_{k}": v for k, v in self.params.items()}

    @property
    def has_input(self) -> bool:
        """Whether this component's behaviour depends on an exogenous input u (e.g. a setpoint)."""
        return False

    @property
    def input_param_name(self) -> Optional[str]:
        """Name of the parameter (within self.params) that the exogenous input u replaces, if has_input."""
        return None


class ConstantPowerLoad(BusComponent):
    """
    A tightly regulated constant-power load, i_inj = -P/v_eff, with the
    same smooth (C1-continuous) voltage floor used in
    `models.dc_microgrid_cpl` to avoid a singularity as v -> 0.
    """

    def __init__(self, id: str, bus: str, P: float, v_floor: float = 0.05):
        super().__init__(id, bus, P=P, v_floor=v_floor)

    def current_injection(self, v_bus: float, local_x: np.ndarray, u: Optional[float] = None) -> float:
        P = self.params["P"] if u is None else u
        v_floor = self.params["v_floor"]
        delta = 0.02
        v_eff = 0.5 * (v_bus + v_floor) + 0.5 * ((v_bus - v_floor) ** 2 + delta ** 2) ** 0.5
        return -P / v_eff

    @property
    def has_input(self) -> bool:
        return True  # P is the natural manifold-continuation / disturbance parameter

    @property
    def input_param_name(self) -> str:
        return "P"


class ConstantImpedanceLoad(BusComponent):
    """A fixed-resistance load, i_inj = -v/R."""

    def __init__(self, id: str, bus: str, R: float):
        super().__init__(id, bus, R=R)

    def current_injection(self, v_bus: float, local_x: np.ndarray, u: Optional[float] = None) -> float:
        R = self.params["R"] if u is None else u
        return -v_bus / R


class ConstantCurrentLoad(BusComponent):
    """A fixed-current load, i_inj = -I."""

    def __init__(self, id: str, bus: str, I: float):
        super().__init__(id, bus, I=I)

    def current_injection(self, v_bus: float, local_x: np.ndarray, u: Optional[float] = None) -> float:
        I = self.params["I"] if u is None else u
        return -I


class IdealSource(BusComponent):
    """
    A Thevenin-equivalent ideal DC source behind a series resistance:
    i_inj = (v_source - v_bus) / R_source. Has no dynamic state of its
    own; contrast with a `network.converter.Converter` built from a
    stateful `ElectricalModel`, which is the more general way to
    represent an actively controlled source (Phase V2.3 onward).
    """

    def __init__(self, id: str, bus: str, v_source: float, R_source: float):
        super().__init__(id, bus, v_source=v_source, R_source=R_source)

    def current_injection(self, v_bus: float, local_x: np.ndarray, u: Optional[float] = None) -> float:
        v_source = self.params["v_source"] if u is None else u
        R_source = self.params["R_source"]
        return (v_source - v_bus) / R_source

    @property
    def has_input(self) -> bool:
        return True

    @property
    def input_param_name(self) -> str:
        return "v_source"
