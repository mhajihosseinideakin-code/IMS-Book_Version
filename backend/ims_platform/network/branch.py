"""
network.branch
---------------

A Branch connects two buses. Every branch in this first version of the
network layer is an R-L line, so that its current is always a genuine
dynamic state (L*di/dt = v_from - v_to - R*i) -- this deliberately avoids
needing an algebraic (DAE) solve for Kirchhoff's voltage law, which
requires either an inductance on every branch or a further reduction the
architecture document's DAE Builder/ODE Reduction subsystems (Part V)
are the natural future home for. Purely resistive branches, transformers,
and switches are follow-up component-library work, not addressed here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Line:
    """
    An R-L line between two buses. Current is defined positive from
    `from_bus` to `to_bus`.
    """
    id: str
    from_bus: str
    to_bus: str
    R: float
    L: float
    i_init: float = 0.0

    def __post_init__(self):
        if self.R < 0:
            raise ValueError(f"Line '{self.id}': R must be non-negative.")
        if self.L <= 0:
            raise ValueError(f"Line '{self.id}': L must be positive.")

    @property
    def state_name(self) -> str:
        return f"i_{self.id}"

    @property
    def param_names(self):
        return (f"R_{self.id}", f"L_{self.id}")
