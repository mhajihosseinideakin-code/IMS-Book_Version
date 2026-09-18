"""
models.converter_cpl_paper
----------------------------

The two-bus DC converter--constant-power-load (CPL) system from
"Intrinsic Manifold Stability and Manifold-Reshaping Control for
Large-Signal Stability of Converter-Dominated DC Systems with
Constant-Power Loads" (Hajihosseini et al.), states x = (i_l, v_b, v_o).

Role of this module
--------------------
`control.mrc_synthesis.MRCSynthesizer` is a generic, model-independent
engine: it derives a control law from whatever symbolic dynamics and
manifold constraint it is given, with no built-in knowledge of converters,
CPLs, or this paper. This module exists solely to give that generic
engine one independently-published validation case to be checked against
(tests/test_mrc_synthesis.py): supplied with this model's dynamics and
manifold constraint, the engine's derived control law is symbolically
identical to the manuscript's published eq. 13. That result validates the
engine; it does not depend on this particular reconstruction being a
complete or numerically well-posed model of the physical converter in
every respect.

Open item on this validation example specifically (not on the Symbolic
Engine or MRCSynthesizer): a closed-loop eigenvalue check of this exact
3-state reconstruction shows a structurally positive tangential mode
(P/(C*v_b^2), independent of parameter choice) at the high-voltage
equilibrium -- i.e. this minimal reconstruction, evaluated on its own
merits as a plant model, is not asymptotically stable as currently
specified, most likely because it omits an inner current-control loop the
manuscript describes but this module does not model. This is recorded as
an open question about this validation fixture; it is not pursued further
here, and it does not bear on the correctness of the symbolic derivation,
which was checked by exact algebraic equality against the published
control law, independent of any closed-loop numerical experiment.

States
------
    i_l : line/inductor current (A)
    v_b : DC bus voltage (V)
    v_o : converter terminal voltage (V) -- itself a state, since the
          outer control loop actuates its derivative: v_o_dot = u.

Input
-----
    u : outer-loop actuation, u = v_o_dot (V/s)

Dynamics (paper eq. 1-2, plus the outer-loop integrator v_o_dot = u under
the paper's fast-inner-loop time-scale-separation assumption, eq. 3 & 16):

    L * i_l_dot = v_o - R*i_l - v_b
    C * v_b_dot = i_l - P/v_b
    v_o_dot     = u

Manifold constraint (paper eq. 6): e_m(x) = v_b - (v_o - R*i_l)
"""

from __future__ import annotations

import numpy as np
from typing import Dict

from ..core.system import DynamicalSystem


class ConverterCPLPaper(DynamicalSystem):
    state_names = ("i_l", "v_b", "v_o")
    input_names = ("u",)

    def default_params(self) -> Dict:
        # Table II of the paper.
        return dict(
            R=0.20,      # line resistance (Ohm)
            L=1.5e-3,    # line inductance (H)
            C=2.5e-3,    # bus capacitance (F)
            P=10e3,      # CPL power (W), nominal / disturbance parameter
        )

    def default_input(self) -> np.ndarray:
        return np.array([0.0])  # u = v_o_dot; zero at a genuine equilibrium

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray, p: Dict) -> np.ndarray:
        i_l, v_b, v_o = x
        uu = u[0]
        di_l = (v_o - p["R"] * i_l - v_b) / p["L"]
        dv_b = (i_l - p["P"] / v_b) / p["C"]
        dv_o = uu
        return np.array([di_l, dv_b, dv_o])

    def admissible(self, x: np.ndarray) -> bool:
        i_l, v_b, v_o = x
        return bool(v_b > 40.0 and abs(i_l) < 500.0)  # generous bounds around a ~400 V nominal bus

    def manifold_residual_numeric(self, x: np.ndarray, p: Dict = None) -> float:
        """Numeric evaluation of e_m(x) = v_b - (v_o - R*i_l) (paper eq. 6), for direct comparison/plots."""
        p = p or self.params
        i_l, v_b, v_o = x
        return v_b - (v_o - p["R"] * i_l)

    # ------------------------------------------------------------------
    # Symbolic contract (core.system.DynamicalSystem), for MRCSynthesizer
    # ------------------------------------------------------------------
    def symbolic_dynamics(self, x_syms, u_syms, p_syms):
        i_l, v_b, v_o = x_syms
        (uu,) = u_syms
        R, L, C, P = p_syms["R"], p_syms["L"], p_syms["C"], p_syms["P"]
        di_l = (v_o - R * i_l - v_b) / L
        dv_b = (i_l - P / v_b) / C
        dv_o = uu
        return [di_l, dv_b, dv_o]

    def symbolic_manifold_constraint(self, x_syms, u_syms, p_syms):
        i_l, v_b, v_o = x_syms
        R = p_syms["R"]
        return v_b - (v_o - R * i_l)
