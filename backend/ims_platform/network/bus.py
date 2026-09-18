"""
network.bus
-----------

A Bus is a network node: either a dynamic DC bus with its own shunt
capacitance (voltage is a state variable, evolved by Kirchhoff's current
law) or an ideal/slack bus with a fixed voltage (no dynamics of its own
-- a boundary condition, such as an ideal DC source).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Bus:
    """
    A DC network node.

    Parameters
    ----------
    id : str
        Unique bus identifier, used to build state/parameter names and
        to reference this bus from branches and attached components.
    C : float, optional
        Shunt capacitance (F). If None, this is an ideal/slack bus with
        a fixed voltage `v_fixed` (a parameter, not a state) -- the
        natural representation of an ideal DC source's terminal.
    v_fixed : float, optional
        Fixed voltage (V), required if C is None.
    v_init : float, optional
        Initial-guess / nominal voltage (V) for a dynamic bus, used to
        seed equilibrium solving and manifold continuation.
    v_min : float
        Admissibility lower bound on this bus's voltage (V).
    """
    id: str
    C: Optional[float] = None
    v_fixed: Optional[float] = None
    v_init: Optional[float] = None
    v_min: float = 0.0

    def __post_init__(self):
        if self.C is None and self.v_fixed is None:
            raise ValueError(f"Bus '{self.id}': an ideal bus (C=None) must specify v_fixed.")
        if self.C is not None and self.C <= 0:
            raise ValueError(f"Bus '{self.id}': capacitance C must be positive if given.")

    @property
    def is_dynamic(self) -> bool:
        """True if this bus has its own voltage state; False if it is an ideal/slack bus."""
        return self.C is not None

    @property
    def state_name(self) -> str:
        return f"v_{self.id}"

    @property
    def param_name(self) -> str:
        return f"C_{self.id}" if self.is_dynamic else f"v_{self.id}_fixed"
