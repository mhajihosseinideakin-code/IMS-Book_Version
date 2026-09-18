"""
models.grid_forming_inverter
-----------------------------

A droop-controlled grid-forming inverter connected through a line
reactance to a stiff voltage bus -- the converter-dominated analogue of
the classical synchronous-machine swing equation, and a natural first
example for IMS because its intrinsic manifold is the classic
saddle-type surface (stable equilibrium / unstable equilibrium pair
per power setpoint) familiar from equal-area / transient-stability
analysis, now reinterpreted through the recoverability lens.

States
------
    delta : power angle (rad) across the line
    omega : frequency deviation from nominal (rad/s), droop-controlled

Input
-----
    P_set : active-power setpoint (p.u.) -- the operating/exogenous
            parameter used to trace the intrinsic manifold.

Dynamics
--------
    ddelta/dt = omega
    domega/dt = (1/tau_p) * ( -omega + m_p * (P_set - P_load - P_e(delta)) )
    P_e(delta) = (E * V / X) * sin(delta)

P_load (default 0.0) is an optional local load drawn at the inverter's
own terminal bus, before the line to the infinite bus -- i.e. power the
inverter must additionally supply beyond what it exports through the
line. Defaulting to 0.0 exactly recovers the original no-load equation,
so this is backward compatible with every existing project, test, and
saved case that doesn't set it: P_load only changes behavior when a
caller deliberately provides a nonzero value.
"""

from __future__ import annotations

import numpy as np
from typing import Dict

from ..core.system import DynamicalSystem


class GridFormingInverter(DynamicalSystem):
    state_names = ("delta", "omega")
    input_names = ("P_set",)

    # Admissible-region bounds (see `admissible` below) exposed as named
    # attributes rather than only inline literals, so callers (e.g. the
    # Explorer's Project Information section) can report the actual
    # enforced envelope rather than maintaining a separate copy of it.
    delta_max_frac_pi: float = 0.95
    omega_max: float = 25.0

    def default_params(self) -> Dict:
        return dict(
            E=1.0,        # inverter internal EMF magnitude (p.u.)
            V=1.0,        # grid bus voltage magnitude (p.u.)
            X=0.3,        # line reactance (p.u.)
            tau_p=0.05,   # droop filter time constant (s)
            m_p=1.0,      # droop gain
            P_load=0.0,   # local load at the inverter's own bus (p.u.); 0.0 = no load, original behavior
        )

    def default_input(self) -> np.ndarray:
        return np.array([0.5])  # nominal P_set (p.u.)

    def electrical_power(self, delta: float) -> float:
        p = self.params
        return (p["E"] * p["V"] / p["X"]) * np.sin(delta)

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray, p: Dict) -> np.ndarray:
        delta, omega = x
        P_set = u[0]
        P_load = p.get("P_load", 0.0)
        P_e = (p["E"] * p["V"] / p["X"]) * np.sin(delta)
        ddelta = omega
        domega = (1.0 / p["tau_p"]) * (-omega + p["m_p"] * (P_set - P_load - P_e))
        return np.array([ddelta, domega])

    def admissible(self, x: np.ndarray) -> bool:
        delta, omega = x
        return bool(abs(delta) < self.delta_max_frac_pi * np.pi and abs(omega) < self.omega_max)
