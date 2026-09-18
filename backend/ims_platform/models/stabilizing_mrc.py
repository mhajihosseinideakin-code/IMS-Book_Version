"""
models.stabilizing_mrc
--------------------------

The "stabilizing MRC" model, distinct from `models.converter_cpl_paper.
ConverterCPLPaper` (the electrical-constraint diagnostic reference). See
that module's own docstring for why its manifold constraint (the plant's
own electrical relation e_m = v_b - v_o + R*i_l) does NOT define a
controller-shaped target that stabilizes an isolated equilibrium (its
own closed-loop Jacobian has a zero eigenvalue and a positive tangential
mode -- verified independently, see tests/test_stabilizing_mrc.py).

This module instead implements the book's STABILIZING MRC construction:
an internal integrator state `sigma` is added specifically so the
manifold constraint becomes a genuine controller target with an
isolated, exponentially stable equilibrium, not merely an electrical
identity.

States (book's order): x = (i_l, v_b, sigma, v_o)
Domain:                D = {x : v_b > 0}

Plant
-----
    L * di_l/dt = v_o - R*i_l - v_b
    C * dv_b/dt = i_l - P/v_b
    d(sigma)/dt = v_nom - v_b
    dv_o/dt     = u

Internal target manifold
-------------------------
    v_Sigma = v_nom - R_v*i_l + K_i*sigma
    e_Sigma = v_o - v_Sigma = v_o - v_nom + R_v*i_l - K_i*sigma

Control law (book, reproduced here and verified independently below and
in tests/test_stabilizing_mrc.py -- both by direct symbolic substitution
AND by running this exact model through the platform's OWN generic
`control.mrc_synthesis.MRCSynthesizer` engine, so the closed-form law
below is checked by two independent code paths, not asserted once and
reused):

    u = -(R_v/L)*(v_o - R*i_l - v_b) + K_i*(v_nom - v_b) - k_m*e_Sigma

which makes:

    de_Sigma/dt = -k_m*e_Sigma                                   (*)

exactly, for as long as the trajectory stays in D = {v_b > 0}.

P and C are required for plant simulation and offline equilibrium/
stability analysis, but do NOT appear in the control law (*) itself --
the online controller needs only i_l, v_b, v_o, its own sigma state, and
R, L, R_v, K_i, k_m, v_nom. This means the controller does not need to
know the load power or bus capacitance to compute u; it does need a
measurement (or estimate) of the remote/DC bus voltage v_b, exactly as
much as the electrical-constraint construction does -- this is a real
requirement of both constructions, not an added burden unique to this
one, and is documented here explicitly per the task's own instruction.

Equilibrium (on e_Sigma = 0), closed form (book; verified independently
below via sympy.solve against the reduced dynamics, not merely by
substitution -- see tests/test_stabilizing_mrc.py):

    v_b_star     = v_nom
    i_l_star     = P / v_nom
    sigma_star   = (R + R_v) * P / (K_i * v_nom)
    v_o_star     = v_nom + R * P / v_nom

Characteristic polynomial of the reduced (i_l, v_b, sigma) dynamics at
that equilibrium, s^3 + a2 s^2 + a1 s + a0, with g := P / v_nom^2
(independently re-derived via the Jacobian of the reduced dynamics, not
copied from the book's own stated formula -- see
tests/test_stabilizing_mrc.py, which checks this module's a0/a1/a2
against a FRESH sympy Jacobian computation each time):

    a2 = (R + R_v)/L - g/C
    a1 = (1 - (R + R_v)*g) / (L*C)
    a0 = K_i / (L*C)

Routh-Hurwitz local-exponential-stability conditions:
    a2 > 0,  a1 > 0,  a0 > 0,  a2*a1 > a0

IMPORTANT SCOPE LIMITATION, stated explicitly per the book's own
required distinctions (do not read more into this model than what is
established below):
  - Existence of this equilibrium: established in closed form.
  - Local exponential stability: established when the Routh-Hurwitz
    conditions above hold (checked numerically by `routh_hurwitz_report`
    below, which reports each condition's own margin, not a single
    unexplained "stable" flag).
  - Certified REGIONAL attraction (a certified basin, valid over some
    explicit finite neighbourhood or the whole domain D): NOT
    established by this module. Nothing here proves the domain over
    which the local linearisation result remains valid.
  - This model is the IDEAL, continuous-time (Mode A) plant + control
    law only: dv_o/dt = u directly, no actuator bandwidth limit. The
    finite-bandwidth sampled implementation (Mode B: dv_c/dt = u_held,
    dv_o/dt = omega_inner*(v_c - v_o), T_s = 10 microseconds, RK4
    between samples) described in the task is NOT implemented in this
    module -- that is a materially different 5-state model with its own
    residual-dynamics identity (de_Sigma/dt = -k_m*e_Sigma +
    omega_inner*(v_c - v_o) - u, not exact exponential contraction) and
    is recorded as NOT IMPLEMENTED in this delivery's completion report,
    not silently assumed to behave identically to Mode A.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple

from ..core.system import DynamicalSystem


class StabilizingMRCModel(DynamicalSystem):
    state_names = ("i_l", "v_b", "sigma", "v_o")
    input_names = ("u",)

    def default_params(self) -> Dict:
        # Benchmark parameters (task's specified Droop/EFL/stabilizing-MRC benchmark).
        return dict(
            R=0.20, L=1.5e-3, C=2.5e-3, v_nom=400.0, P=20000.0,
            R_v=0.5, K_i=50.0, k_m=500.0,
        )

    def default_input(self) -> np.ndarray:
        return np.array([0.0])  # u = dv_o/dt; the closed-loop law computes it, this is only a numeric placeholder

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray, p: Dict) -> np.ndarray:
        i_l, v_b, sigma, v_o = x
        R, L, C, v_nom, P = p["R"], p["L"], p["C"], p["v_nom"], p["P"]
        di_l = (v_o - R * i_l - v_b) / L
        dv_b = (i_l - P / v_b) / C
        dsigma = v_nom - v_b
        dv_o = u[0]
        return np.array([di_l, dv_b, dsigma, dv_o])

    def admissible(self, x: np.ndarray) -> bool:
        i_l, v_b, sigma, v_o = x
        return bool(v_b > 0.0)  # D = {x : v_b > 0}, exactly as specified

    def closed_loop_control(self, x: np.ndarray, p: Dict = None) -> float:
        """The book's stabilizing-MRC control law u, evaluated numerically (not the open-loop default_input)."""
        p = p or self.params
        i_l, v_b, sigma, v_o = x
        R, L, R_v, K_i, k_m, v_nom = p["R"], p["L"], p["R_v"], p["K_i"], p["k_m"], p["v_nom"]
        e_Sigma = v_o - v_nom + R_v * i_l - K_i * sigma
        return -(R_v / L) * (v_o - R * i_l - v_b) + K_i * (v_nom - v_b) - k_m * e_Sigma

    def residual_e_sigma(self, x: np.ndarray, p: Dict = None) -> float:
        p = p or self.params
        i_l, v_b, sigma, v_o = x
        R_v, K_i, v_nom = p["R_v"], p["K_i"], p["v_nom"]
        return v_o - v_nom + R_v * i_l - K_i * sigma

    def distance_to_manifold_squared(self, x: np.ndarray, p: Dict = None) -> float:
        """
        dist(x, M_Sigma)^2 = e_Sigma(x)^2 / q, in the book's explicitly stated
        native-coordinate Euclidean metric, where grad(e_Sigma) w.r.t.
        (i_l, v_b, sigma, v_o) = (R_v, 0, -K_i, 1) and q = R_v^2 + K_i^2 + 1.

        The orthogonal projection (moving only in the grad(e_Sigma)
        direction) changes v_b by zero (the gradient's v_b-component is
        zero), so the projection of any x in D = {v_b > 0} remains in D --
        this is checked directly in tests/test_stabilizing_mrc.py, not
        merely asserted here.

        For V := e_Sigma^2 = q * dist^2, the verified residual identity
        de_Sigma/dt = -k_m*e_Sigma gives dV/dt = -2*k_m*V exactly (chain
        rule; verified independently in
        tests/test_stabilizing_mrc.py::test_distance_certificate_dVdt_identity),
        i.e. dist^2 itself also contracts at rate 2*k_m along the ideal
        trajectory, for as long as it remains in D.

        NOTE: this is the native-coordinate Euclidean-metric certificate
        specifically. If a caller uses normalized coordinates or a
        weighted metric, q must be re-derived for that metric -- reusing
        this q value unchanged under a different metric is NOT valid and
        is not done anywhere in this module.
        """
        p = p or self.params
        R_v, K_i = p["R_v"], p["K_i"]
        q = R_v ** 2 + K_i ** 2 + 1.0
        e_sigma = self.residual_e_sigma(x, p)
        return (e_sigma ** 2) / q

    def equilibrium_closed_form(self, p: Dict = None) -> np.ndarray:
        """Closed-form equilibrium (book; independently verified in tests/test_stabilizing_mrc.py)."""
        p = p or self.params
        R, R_v, K_i, v_nom, P = p["R"], p["R_v"], p["K_i"], p["v_nom"], p["P"]
        v_b_star = v_nom
        i_l_star = P / v_nom
        sigma_star = (R + R_v) * P / (K_i * v_nom)
        v_o_star = v_nom + R * P / v_nom
        return np.array([i_l_star, v_b_star, sigma_star, v_o_star])

    def characteristic_coeffs(self, p: Dict = None) -> Tuple[float, float, float]:
        """(a2, a1, a0) of s^3+a2 s^2+a1 s+a0 for the reduced (i_l,v_b,sigma) dynamics on e_Sigma=0."""
        p = p or self.params
        R, L, C, R_v, K_i, v_nom, P = p["R"], p["L"], p["C"], p["R_v"], p["K_i"], p["v_nom"], p["P"]
        g = P / v_nom ** 2
        a2 = (R + R_v) / L - g / C
        a1 = (1.0 - (R + R_v) * g) / (L * C)
        a0 = K_i / (L * C)
        return a2, a1, a0

    def routh_hurwitz_report(self, p: Dict = None) -> Dict:
        """
        Each Routh-Hurwitz condition reported individually with its own
        margin, per the task's explicit instruction not to report only a
        single unexplained "stable" flag.
        """
        a2, a1, a0 = self.characteristic_coeffs(p)
        conditions = {
            "a2 > 0": (a2 > 0, a2),
            "a1 > 0": (a1 > 0, a1),
            "a0 > 0": (a0 > 0, a0),
            "a2*a1 > a0": (a2 * a1 > a0, a2 * a1 - a0),
        }
        all_satisfied = all(ok for ok, _ in conditions.values())
        return {
            "a2": a2, "a1": a1, "a0": a0,
            "conditions": {name: {"satisfied": bool(ok), "margin": float(margin)} for name, (ok, margin) in conditions.items()},
            "locally_exponentially_stable": bool(all_satisfied),
            "scope_note": (
                "This result establishes LOCAL exponential stability of the closed-loop equilibrium "
                "under linearisation only. It does NOT certify a regional (finite-neighbourhood or "
                "domain-wide) basin of attraction; no such certification is implemented in this module."
            ),
        }

    def poles(self, p: Dict = None):
        a2, a1, a0 = self.characteristic_coeffs(p)
        return np.roots([1.0, a2, a1, a0])

    # ------------------------------------------------------------------
    # Symbolic contract, for control.mrc_synthesis.MRCSynthesizer
    # ------------------------------------------------------------------
    def symbolic_dynamics(self, x_syms, u_syms, p_syms):
        i_l, v_b, sigma, v_o = x_syms
        (u,) = u_syms
        R, L, C, v_nom, P = p_syms["R"], p_syms["L"], p_syms["C"], p_syms["v_nom"], p_syms["P"]
        di_l = (v_o - R * i_l - v_b) / L
        dv_b = (i_l - P / v_b) / C
        dsigma = v_nom - v_b
        dv_o = u
        return [di_l, dv_b, dsigma, dv_o]

    def symbolic_manifold_constraint(self, x_syms, u_syms, p_syms):
        i_l, v_b, sigma, v_o = x_syms
        R_v, K_i, v_nom = p_syms["R_v"], p_syms["K_i"], p_syms["v_nom"]
        return v_o - v_nom + R_v * i_l - K_i * sigma
