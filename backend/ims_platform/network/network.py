"""
network.network
----------------

`Network`: a topology (buses + branches) with components bound to buses
-- the user-facing input to the Automatic Model Builder (architecture
document Part IV Section 4.1/4.2). This is deliberately the layer a user
or IMS Studio works at, per the component hierarchy in that section:
Network -> Component -> ... -> Assembler -> DynamicSystem.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .bus import Bus
from .branch import Line
from .components import BusComponent


class NetworkValidationError(ValueError):
    """Raised when a Network's topology or bindings are inconsistent."""


class Network:
    """
    A DC network topology: a set of buses, a set of R-L line branches
    connecting them, and a set of components bound to buses.
    """

    def __init__(self, name: str = "network"):
        self.name = name
        self.buses: Dict[str, Bus] = {}
        self.branches: List[Line] = []
        self.components: List[BusComponent] = []

    # ------------------------------------------------------------------
    def add_bus(self, bus: Bus) -> "Network":
        if bus.id in self.buses:
            raise NetworkValidationError(f"duplicate bus id '{bus.id}'")
        self.buses[bus.id] = bus
        return self

    def add_line(self, line: Line) -> "Network":
        for end in (line.from_bus, line.to_bus):
            if end not in self.buses:
                raise NetworkValidationError(f"Line '{line.id}' references unknown bus '{end}'")
        self.branches.append(line)
        return self

    def add_component(self, component: BusComponent) -> "Network":
        if component.bus not in self.buses:
            raise NetworkValidationError(f"Component '{component.id}' references unknown bus '{component.bus}'")
        self.components.append(component)
        return self

    # ------------------------------------------------------------------
    def validate(self) -> None:
        """
        Structural checks beyond the incremental ones in add_*: every
        dynamic bus must have at least one incident branch or component
        (otherwise its voltage state has no dynamics at all), and the
        network must have at least one bus.
        """
        if not self.buses:
            raise NetworkValidationError("network has no buses")

        incident = {bus_id: 0 for bus_id in self.buses}
        for line in self.branches:
            incident[line.from_bus] += 1
            incident[line.to_bus] += 1
        for comp in self.components:
            incident[comp.bus] += 1

        for bus_id, bus in self.buses.items():
            if bus.is_dynamic and incident[bus_id] == 0:
                raise NetworkValidationError(
                    f"dynamic bus '{bus_id}' has no incident branch or component; "
                    "its voltage would have no dynamics."
                )

    def dynamic_buses(self) -> List[Bus]:
        return [b for b in self.buses.values() if b.is_dynamic]

    def neighbors(self, bus_id: str) -> List[str]:
        """Adjacent bus ids (graph view, Part IV Section 4.3: Network -> Graph)."""
        out = []
        for line in self.branches:
            if line.from_bus == bus_id:
                out.append(line.to_bus)
            elif line.to_bus == bus_id:
                out.append(line.from_bus)
        return out

    def components_at(self, bus_id: str) -> List[BusComponent]:
        return [c for c in self.components if c.bus == bus_id]

    def find_input_component(self, component_id: Optional[str] = None) -> Optional[BusComponent]:
        """
        Locate the component to expose as the assembled system's single
        exogenous input (for manifold continuation / disturbance
        studies), matching the single-parameter continuation the
        existing IntrinsicManifold / RecoverabilityAnalyzer machinery
        expects (architecture document Section 5.4 notes multi-parameter
        continuation as future work). Defaults to the first component
        with has_input=True.
        """
        if component_id is not None:
            matches = [c for c in self.components if c.id == component_id]
            if not matches:
                raise NetworkValidationError(f"no component with id '{component_id}'")
            return matches[0]
        for c in self.components:
            if c.has_input:
                return c
        return None
