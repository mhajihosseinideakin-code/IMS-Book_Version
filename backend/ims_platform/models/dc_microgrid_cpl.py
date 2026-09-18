"""
models.dc_microgrid_cpl
------------------------

A DC microgrid bus (grid-forming converter + line inductance/resistance)
feeding a downstream tightly-regulated constant-power load (CPL) -- the
textbook example of large-signal, converter-dominated instability that
small-signal analysis alone can badly mis-characterise: the system has
TWO equilibria for a given load power (one stable, one a saddle acting
as the literal recoverability boundary in this simple 2-state case),
and whether a disturbance recovers depends entirely on which side of
the unstable equilibrium the post-disturbance state lands on.

States
------
    i : inductor / line current (p.u.)
    v : DC bus (capacitor) voltage (p.u.)

Input
-----
    P_load : constant-power load demand (p.u.) -- the operating
             parameter used to trace the intrinsic manifold.

Dynamics
--------
    L di/dt = Vin - r*i - v
    C dv/dt = i - P_load / v
"""

from __future__ import annotations

import numpy as np
from typing import Dict

from ..core.system import DynamicalSystem


class DCMicrogridCPL(DynamicalSystem):
    state_names = ("i", "v")
    input_names = ("P_load",)

    def default_params(self) -> Dict:
        # Chosen so that, for moderate CPL power, the high-voltage
        # equilibrium is a genuine stable focus and the low-voltage
        # equilibrium is a genuine saddle (real eigenvalues of opposite
        # sign) -- i.e. enough damping (r/L) relative to the CPL's
        # incremental negative admittance (~P/(C v^2)) per the classical
        # Middlebrook stability criterion for CPL-fed converters.
        return dict(
            L=0.005,     # line/converter inductance (H, normalised)
            C=0.05,      # DC bus capacitance (F, normalised)
            r=0.15,      # line resistance / damping (p.u.)
            Vin=1.2,     # source-side regulated voltage (p.u.)
            v_floor=0.05,  # numerical floor to avoid singularity at v -> 0
        )

    def default_input(self) -> np.ndarray:
        return np.array([0.5])  # nominal P_load (p.u.)

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray, p: Dict) -> np.ndarray:
        i, v = x
        P_load = u[0]
        # Smooth ("soft-max") floor on the effective voltage used for the
        # CPL's 1/v current draw: v_eff -> v for v >> v_floor, v_eff ->
        # v_floor for v <= v_floor, with a C1-continuous transition. A hard
        # sign-based clamp here creates a discontinuous derivative right at
        # the operating region a collapsing trajectory passes through,
        # which makes adaptive ODE solvers take pathologically many tiny
        # steps (or effectively hang) -- the smooth version keeps the
        # physics (a bounded CPL current draw at low voltage) while
        # remaining numerically well-behaved.
        delta = 0.02
        v_eff = 0.5 * (v + p["v_floor"]) + 0.5 * np.sqrt((v - p["v_floor"]) ** 2 + delta ** 2)
        di = (p["Vin"] - p["r"] * i - v) / p["L"]
        dv = (i - P_load / v_eff) / p["C"]
        return np.array([di, dv])

    def admissible(self, x: np.ndarray) -> bool:
        i, v = x
        return bool(v > 0.3 and abs(i) < 15.0)

    def equilibria_analytic(self, P_load: float) -> np.ndarray:
        """
        Closed-form equilibria for validation: v* solves
        v^2 - Vin*v + P_load*r = 0  =>  v* = (Vin +/- sqrt(Vin^2 - 4 P r)) / 2
        Returns array of shape (n_eq, 2) with columns [i*, v*]; empty if none real.
        """
        p = self.params
        disc = p["Vin"] ** 2 - 4 * P_load * p["r"]
        if disc < 0:
            return np.empty((0, 2))
        sq = np.sqrt(disc)
        v_hi = (p["Vin"] + sq) / 2
        v_lo = (p["Vin"] - sq) / 2
        out = []
        for v_star in (v_hi, v_lo):
            if v_star > 1e-6:
                i_star = P_load / v_star
                out.append([i_star, v_star])
        return np.array(out)
