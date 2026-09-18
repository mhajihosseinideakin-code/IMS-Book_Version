"""
network.topology
------------------

`Bus`, `Branch`, and `Network`: the plain-data description of a DC network
topology (architecture document Part IV §4.1's "Digital Network" stage,
and §4.2's topology-ingestion job of the Automatic Model Builder).

A `Network` is exactly what a user (or IMS Studio, eventually) provides:
a set of buses, a set of branches connecting them, and which component
(if any) is attached to each bus. Nothing here knows how to turn this
into a `DynamicalSystem` -- that is `network.builder.AutomaticModelBuilder`'s
job, kept deliberately separate so the topology description stays plain
data with no dynamics-assembly logic embedded in it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .components import DCComponent


class NetworkValidationError(ValueError):
    """Raised when a Network's topology is malformed (e.g. a dangling branch, no slack bus)."""


@dataclass
class Bus:
    """
    A network node. `bus_type` is either:
        "slack"   -- fixed voltage (a parameter, not a dynamic state);
                     represents a stiff source / infinite bus.
        "dynamic" -- has its own capacitance; voltage is a state governed
                     by Kirchhoff's current law (sum of branch currents +
                     any attached component's injection).
    """
    id: str
    bus_type: str  # "slack" | "dynamic"
    voltage: Optional[float] = None       # required if bus_type == "slack"
    capacitance: Optional[float] = None   # required if bus_type == "dynamic"
    component: Optional[DCComponent] = None

    def __post_init__(self):
        if self.bus_type not in ("slack", "dynamic"):
            raise NetworkValidationError(f"Bus '{self.id}': bus_type must be 'slack' or 'dynamic', got '{self.bus_type}'")
        if self.bus_type == "slack" and self.voltage is None:
            raise NetworkValidationError(f"Bus '{self.id}': slack buses require a fixed 'voltage'.")
        if self.bus_type == "dynamic" and self.capacitance is None:
            raise NetworkValidationError(f"Bus '{self.id}': dynamic buses require a 'capacitance'.")


@dataclass
class Branch:
    """An R-L line/branch connecting two buses. Branch current is a dynamic state:
    L * di/dt = v_from - v_to - R*i  (positive i flows from `from_bus` to `to_bus`)."""
    id: str
    from_bus: str
    to_bus: str
    resistance: float
    inductance: float


@dataclass
class Network:
    """A complete DC network topology: buses, branches, and their bound components."""
    buses: List[Bus] = field(default_factory=list)
    branches: List[Branch] = field(default_factory=list)
    name: str = "network"

    def __post_init__(self):
        self._bus_index = {b.id: b for b in self.buses}
        self._validate()

    def _validate(self):
        if len(self._bus_index) != len(self.buses):
            raise NetworkValidationError("Duplicate bus ids in network.")
        branch_ids = [br.id for br in self.branches]
        if len(set(branch_ids)) != len(branch_ids):
            raise NetworkValidationError("Duplicate branch ids in network.")
        for br in self.branches:
            if br.from_bus not in self._bus_index:
                raise NetworkValidationError(f"Branch '{br.id}' references unknown bus '{br.from_bus}'.")
            if br.to_bus not in self._bus_index:
                raise NetworkValidationError(f"Branch '{br.id}' references unknown bus '{br.to_bus}'.")
            if br.resistance <= 0 or br.inductance <= 0:
                raise NetworkValidationError(f"Branch '{br.id}': resistance and inductance must be positive.")
        if not any(b.bus_type == "slack" for b in self.buses):
            raise NetworkValidationError(
                "Network has no slack bus. A DC network needs at least one fixed-voltage "
                "reference bus for the system to be well-posed (there is otherwise no "
                "absolute voltage reference)."
            )
        dynamic_buses = [b for b in self.buses if b.bus_type == "dynamic"]
        if not dynamic_buses:
            raise NetworkValidationError("Network has no dynamic buses; nothing to simulate.")
        # connectivity check: every bus reachable from some slack bus via branches
        adjacency: Dict[str, List[str]] = {b.id: [] for b in self.buses}
        for br in self.branches:
            adjacency[br.from_bus].append(br.to_bus)
            adjacency[br.to_bus].append(br.from_bus)
        slack_ids = [b.id for b in self.buses if b.bus_type == "slack"]
        seen = set()
        stack = list(slack_ids)
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(adjacency[cur])
        unreachable = set(self._bus_index) - seen
        if unreachable:
            raise NetworkValidationError(f"Bus(es) {sorted(unreachable)} are not connected to any slack bus.")

    def bus(self, bus_id: str) -> Bus:
        return self._bus_index[bus_id]

    @property
    def dynamic_buses(self) -> List[Bus]:
        return [b for b in self.buses if b.bus_type == "dynamic"]

    @property
    def slack_buses(self) -> List[Bus]:
        return [b for b in self.buses if b.bus_type == "slack"]
