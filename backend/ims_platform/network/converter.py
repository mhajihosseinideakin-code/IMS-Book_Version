"""
network.converter
--------------------

`Converter`: composes an `ElectricalModel` and a `Controller`
(architecture document Part IV Section 4.4) into a single, stateful
`BusComponent` the Automatic Model Builder can attach to a bus like any
other component. This is the class that makes "swap the controller
without touching the converter, swap the topology without touching the
controller" an actual, mechanical operation: `Converter` itself contains
no physics and no control logic at all -- it only wires the two
together and manages the bookkeeping (local state ordering, parameter
namespacing) needed to fit them into the platform's single global state
vector.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from .components import BusComponent
from .electrical_model import ElectricalModel
from .controller import Controller


class Converter(BusComponent):
    """
    A bus-attached, stateful device built from an ElectricalModel and a
    Controller. Ports (current_injection), States (local_state_names),
    and Parameters (prefixed_params) are all derived automatically from
    the two constituent objects -- this class adds no physics of its own.
    """

    def __init__(self, id: str, bus: str, electrical_model: ElectricalModel, controller: Controller):
        super().__init__(id, bus)  # BusComponent.__init__(id, bus, **params); no top-level params here
        self.electrical_model = electrical_model
        self.controller = controller

        self._n_em = len(electrical_model.state_names)
        self._n_ctrl = len(controller.state_names)
        self.n_local_states = self._n_em + self._n_ctrl
        self._local_state_names = tuple(f"em_{n}" for n in electrical_model.state_names) + tuple(
            f"ctrl_{n}" for n in controller.state_names
        )

        # Own parameters are the union of both halves', namespaced so
        # they can never collide with each other or with any other
        # component (Converter.prefixed_params then adds the id prefix
        # on top of this, matching every other BusComponent).
        self.params = {}
        for k, v in electrical_model.params.items():
            self.params[f"em_{k}"] = v
        for k, v in controller.params.items():
            self.params[f"ctrl_{k}"] = v

    # ------------------------------------------------------------------
    def _split(self, local_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return local_x[: self._n_em], local_x[self._n_em:]

    def current_injection(self, v_bus: float, local_x: np.ndarray, u: Optional[float] = None) -> float:
        elec_x, ctrl_x = self._split(local_x)
        signal = self.controller.control_signal(v_bus, elec_x, ctrl_x, u, self.controller.params)
        return self.electrical_model.port_current(v_bus, elec_x, signal, self.electrical_model.params)

    def local_dynamics(self, v_bus: float, local_x: np.ndarray, u: Optional[float] = None) -> np.ndarray:
        elec_x, ctrl_x = self._split(local_x)
        signal = self.controller.control_signal(v_bus, elec_x, ctrl_x, u, self.controller.params)
        d_em = self.electrical_model.local_dynamics(v_bus, elec_x, signal, self.electrical_model.params)
        d_ctrl = self.controller.local_dynamics(v_bus, elec_x, ctrl_x, u, self.controller.params)
        return np.concatenate([d_em, d_ctrl])

    def initial_state_guess(self) -> np.ndarray:
        return np.concatenate([self.electrical_model.initial_state_guess(), self.controller.initial_state_guess()])

    @property
    def has_input(self) -> bool:
        return any(k in self.controller.param_names for k in ("v_ref", "v_nom", "d"))

    @property
    def input_param_name(self) -> Optional[str]:
        if "v_ref" in self.controller.param_names:
            return "ctrl_v_ref"
        if "v_nom" in self.controller.param_names:
            return "ctrl_v_nom"
        if "d" in self.controller.param_names:
            return "ctrl_d"
        return None
