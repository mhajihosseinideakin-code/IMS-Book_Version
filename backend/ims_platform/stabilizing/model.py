"""
ims_platform.stabilizing.model
-------------------------------

A closed-loop wrapper around the verified open-loop
`StabilizingMRCModel`. The open-loop model has dv_o/dt = u (u exogenous).
This wrapper substitutes the book's stabilizing-MRC control law
u = closed_loop_control(x) INTO the vector field, giving a genuine
autonomous 4-state closed-loop system

    dx/dt = f_cl(x; p)

so that:
  * the inherited generic finite-difference `jacobian(x)` produces the
    full 4x4 CLOSED-LOOP Jacobian numerically (an independent route from
    the analytic expression in `analytic_closed_loop_jacobian`), and
  * `Simulator` can integrate it directly.

No mathematics from the reference is re-derived here; the control law,
residual and equilibrium all come from the parent model.
"""

from __future__ import annotations

import numpy as np
from typing import Dict

from ..models.stabilizing_mrc import StabilizingMRCModel


class StabilizingMRCClosedLoop(StabilizingMRCModel):
    """Autonomous closed-loop form: dv_o/dt = u_law(x)."""

    name_id = "stabilizing_mrc_closed_loop"

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray, p: Dict) -> np.ndarray:
        # Ignore exogenous u; substitute the verified stabilizing control law.
        u_law = self.closed_loop_control(x, p)
        i_l, v_b, sigma, v_o = x
        R, L, C, v_nom, P = p["R"], p["L"], p["C"], p["v_nom"], p["P"]
        di_l = (v_o - R * i_l - v_b) / L
        dv_b = (i_l - P / v_b) / C
        dsigma = v_nom - v_b
        dv_o = u_law
        return np.array([di_l, dv_b, dsigma, dv_o])

    def analytic_closed_loop_jacobian(self, x: np.ndarray, p: Dict = None) -> np.ndarray:
        """
        Closed-form 4x4 Jacobian d f_cl / d x, in state order
        (i_l, v_b, sigma, v_o). Derived by hand from the closed-loop ODE
        (Appendix A). This is the DISPLAYED analytic result; it is
        cross-checked against the independent finite-difference route
        (inherited `jacobian`) in the workflow and in the tests.
        """
        p = p or self.params
        R, L, C, R_v, K_i, k_m = p["R"], p["L"], p["C"], p["R_v"], p["K_i"], p["k_m"]
        i_l, v_b, sigma, v_o = x
        P = p["P"]
        # d(dv_b)/dv_b = d/dv_b[(i_l - P/v_b)/C] = (P / v_b^2) / C
        dvb_dvb = (P / v_b ** 2) / C
        J = np.array([
            [-R / L,               -1.0 / L,          0.0,        1.0 / L],
            [1.0 / C,              dvb_dvb,           0.0,        0.0],
            [0.0,                  -1.0,              0.0,        0.0],
            [R_v * R / L - k_m * R_v,
                                   R_v / L - K_i,     k_m * K_i,  -R_v / L - k_m],
        ])
        return J
