"""
network.assembler
-------------------

The Automatic Model Builder (architecture document Part IV Section 4.2):
takes a `Network` (a topology with bound components) and emits a single
object satisfying the existing `DynamicalSystem` contract, so that every
downstream capability -- equilibrium solving, manifold tracing,
recoverability assessment, MRC -- works on the assembled network exactly
as it already works on a hand-written model, unmodified.

State-space construction
-------------------------
The global state vector is, in order:

    [dynamic bus voltages]  +  [branch currents]  +  [each stateful
    component's own local states, in network.components order, prefixed
    by that component's id]

No algebraic (DAE) elimination is needed in this version: every branch
carries an inductance (network.branch.Line), so Kirchhoff's voltage law
around each branch is already an ODE, and every component (whether
algebraic, like a load, or stateful, like a Converter -- network.converter)
exposes a uniform current_injection/local_dynamics interface regardless
of which kind it is, so the bus KCL sum and the state-vector assembly
below never need to know the difference.

    C_b * dv_b/dt = sum_{lines l incident to b} (+-) i_l  +  sum_{components at b} i_inj
    L_l * di_l/dt = v_(from of l) - v_(to of l) - R_l * i_l
    d(local_x_c)/dt = c.local_dynamics(v_bus_of_c, local_x_c, u_c)   for each stateful component c
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

from ..core.system import DynamicalSystem
from .network import Network


class AssembledNetworkSystem(DynamicalSystem):
    """
    A `DynamicalSystem` automatically assembled from a `Network`. Not
    intended to be constructed directly -- use
    `AutomaticModelBuilder.build(network)`.
    """

    def __init__(self, network: Network, input_component_id: Optional[str] = None, name: Optional[str] = None):
        network.validate()
        self.network = network

        self._dynamic_buses = network.dynamic_buses()
        self._bus_index = {b.id: i for i, b in enumerate(self._dynamic_buses)}
        n_bus_states = len(self._dynamic_buses)

        self._branches = list(network.branches)
        self._branch_index = {ln.id: n_bus_states + i for i, ln in enumerate(self._branches)}
        n_after_branches = n_bus_states + len(self._branches)

        # Allocate a contiguous state slice for every component that has
        # local states (Converter and any future stateful component);
        # stateless components (n_local_states == 0) get no slice.
        self._component_slice: Dict[str, Tuple[int, int]] = {}
        state_names: List[str] = [b.state_name for b in self._dynamic_buses] + [ln.state_name for ln in self._branches]
        cursor = n_after_branches
        for comp in network.components:
            if comp.n_local_states > 0:
                self._component_slice[comp.id] = (cursor, comp.n_local_states)
                state_names.extend(list(comp.local_state_names()))
                cursor += comp.n_local_states
        self.state_names = tuple(state_names)

        self._input_component = network.find_input_component(input_component_id)
        self.input_names = (f"{self._input_component.id}_input",) if self._input_component else ()

        super().__init__(params=self._collect_params(), name=name or network.name)

    # ------------------------------------------------------------------
    def _collect_params(self) -> Dict:
        p: Dict = {}
        for bus in self.network.buses.values():
            if bus.is_dynamic:
                p[f"C_{bus.id}"] = bus.C
            else:
                p[f"v_{bus.id}_fixed"] = bus.v_fixed
        for line in self._branches:
            p[f"R_{line.id}"] = line.R
            p[f"L_{line.id}"] = line.L
        for comp in self.network.components:
            p.update(comp.prefixed_params())
        return p

    def default_params(self) -> Dict:
        # populated in __init__ before super().__init__ is called via
        # `params=` override; DynamicalSystem.__init__ calls this first,
        # so return an empty dict here and let the explicit params= win.
        return {}

    def default_input(self) -> np.ndarray:
        if not self._input_component:
            return np.zeros(0)
        key = f"{self._input_component.id}_{self._input_component.input_param_name}"
        return np.array([self.params.get(key, 0.0)])

    def _bus_voltage(self, bus_id: str, x: np.ndarray) -> float:
        bus = self.network.buses[bus_id]
        if bus.is_dynamic:
            return x[self._bus_index[bus_id]]
        return self.params[f"v_{bus_id}_fixed"]

    def _component_local_x(self, comp_id: str, x: np.ndarray) -> np.ndarray:
        if comp_id not in self._component_slice:
            return np.zeros(0)
        start, n = self._component_slice[comp_id]
        return x[start:start + n]

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray, p: Dict) -> np.ndarray:
        n = self.n_states
        dxdt = np.zeros(n)

        # Kirchhoff's current law at each dynamic bus.
        for bus in self._dynamic_buses:
            idx = self._bus_index[bus.id]
            C = p[f"C_{bus.id}"]
            total = 0.0
            for line in self._branches:
                if line.from_bus == bus.id:
                    total -= x[self._branch_index[line.id]]
                elif line.to_bus == bus.id:
                    total += x[self._branch_index[line.id]]
            for comp in self.network.components_at(bus.id):
                v_here = self._bus_voltage(bus.id, x)
                local_x = self._component_local_x(comp.id, x)
                u_val = None
                if self._input_component is not None and comp.id == self._input_component.id:
                    u_val = float(u[0]) if len(u) else None
                total += comp.current_injection(v_here, local_x, u_val)
            dxdt[idx] = total / C

        # Kirchhoff's voltage law around each branch.
        for line in self._branches:
            idx = self._branch_index[line.id]
            R = p[f"R_{line.id}"]
            L = p[f"L_{line.id}"]
            v_from = self._bus_voltage(line.from_bus, x)
            v_to = self._bus_voltage(line.to_bus, x)
            i = x[idx]
            dxdt[idx] = (v_from - v_to - R * i) / L

        # Each stateful component's own local dynamics.
        for comp in self.network.components:
            if comp.id not in self._component_slice:
                continue
            start, n_local = self._component_slice[comp.id]
            v_here = self._bus_voltage(comp.bus, x)
            local_x = x[start:start + n_local]
            u_val = None
            if self._input_component is not None and comp.id == self._input_component.id:
                u_val = float(u[0]) if len(u) else None
            dxdt[start:start + n_local] = comp.local_dynamics(v_here, local_x, u_val)

        return dxdt

    def admissible(self, x: np.ndarray) -> bool:
        for bus in self._dynamic_buses:
            if x[self._bus_index[bus.id]] < bus.v_min:
                return False
        return True

    def initial_guess(self) -> np.ndarray:
        """A reasonable state-vector initial guess for equilibrium solving, from bus/line/component defaults."""
        x0 = np.zeros(self.n_states)
        for bus in self._dynamic_buses:
            x0[self._bus_index[bus.id]] = bus.v_init if bus.v_init is not None else 1.0
        for line in self._branches:
            x0[self._branch_index[line.id]] = line.i_init
        for comp in self.network.components:
            if comp.id in self._component_slice:
                start, n_local = self._component_slice[comp.id]
                if hasattr(comp, "initial_state_guess"):
                    x0[start:start + n_local] = comp.initial_state_guess()
                else:
                    # Seed local states at 1.0 for voltage-like states, 0.0
                    # otherwise; a component can override by exposing its
                    # own initial guess (see Converter.initial_state_guess).
                    # Simple, workable default for component types that
                    # don't define one.
                    x0[start:start + n_local] = 1.0
        return x0


class AutomaticModelBuilder:
    """
    The Automatic Model Builder (architecture document Part IV Section 4.2):
    the single entry point that turns a `Network` into a `DynamicalSystem`.
    """

    @staticmethod
    def build(network: Network, input_component_id: Optional[str] = None, name: Optional[str] = None) -> AssembledNetworkSystem:
        """
        Assemble `network` into a DynamicalSystem. Raises
        NetworkValidationError if the topology is inconsistent (e.g. a
        dynamic bus with no incident branch or component).
        """
        return AssembledNetworkSystem(network, input_component_id=input_component_id, name=name)
