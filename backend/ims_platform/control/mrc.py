"""
control.mrc
-----------

Implements **manifold-scheduled LQR control**: a conventional (linear-
quadratic) feedback controller whose gain is scheduled over samples of
the intrinsic manifold M, so the linearisation point tracks the system's
current operating region rather than a single fixed setpoint.

.. important::
    Architecture-document correction (Version 3, Part III §3.3): this
    class was previously named ``ManifoldReshapingControl``. That name
    conflated it with Manifold-Reshaping Control (MRC) proper, which is a
    distinct, nonlinear, manifold-residual-based synthesis method (see
    ``control.mrc_synthesis.MRCSynthesizer``) with no linearisation or
    optimal-control step anywhere in its derivation. This class belongs
    to the **Conventional Control Library**, not the IMS Control
    Framework: it is manifold-*aware* (the gain is scheduled by proximity
    to M), but it is still an LQR design, not MRC. The class is renamed
    to ``ScheduledLQRControl`` accordingly; ``ManifoldReshapingControl``
    remains available as a deprecated alias for backward compatibility.

Design approach
----------------
At each control evaluation:
    1. Locate the nearest manifold sample x*(t) to the current state.
    2. Linearise the plant locally about x* (via the platform's generic
       Jacobian utilities) to get (A, B).
    3. Solve the continuous-time algebraic Riccati equation for a local
       LQR gain K(x*) trading off state deviation from M against control
       effort (Q, R design weights).
    4. Apply u = u_ff(x*) + saturate(-K (x - x*)).

Because the gain is recomputed from the *local* manifold target rather
than a single fixed equilibrium, the controller remains effective across
the whole operating range traced by the manifold -- distinguishing it
from a conventional single-point LQR regulator, even though it remains,
architecturally, a conventional (linearisation- and Riccati-based)
design rather than an IMS-native one.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional

from scipy.linalg import solve_continuous_are

from ..core.system import DynamicalSystem
from ..ims.manifold import IntrinsicManifold


@dataclass
class MRCDesignParams:
    Q: np.ndarray            # state-deviation-from-manifold penalty
    R: np.ndarray            # control-effort penalty
    u_min: Optional[np.ndarray] = None
    u_max: Optional[np.ndarray] = None


class ScheduledLQRControl:
    """
    Manifold-scheduled LQR feedback controller (Conventional Control
    Library). See module docstring for the naming correction relative to
    earlier versions of this platform.
    """

    def __init__(
        self,
        system: DynamicalSystem,
        manifold: IntrinsicManifold,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        u_min: Optional[np.ndarray] = None,
        u_max: Optional[np.ndarray] = None,
        gain_refresh_every: int = 1,
    ):
        n = system.n_states
        m = len(system.input_names) if system.input_names else system.default_input().size
        self.system = system
        self.manifold = manifold
        self.params = MRCDesignParams(
            Q=Q if Q is not None else np.eye(n),
            R=R if R is not None else 0.1 * np.eye(max(m, 1)),
            u_min=u_min,
            u_max=u_max,
        )
        self._m = max(m, 1)
        self._call_count = 0
        self.gain_refresh_every = gain_refresh_every
        self._last_K = None
        self._last_target = None
        self._gain_schedule: Optional[list] = None  # precomputed K_i per manifold point

    # ------------------------------------------------------------------
    def _numeric_B(self, x_star: np.ndarray, u0: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        n = self.system.n_states
        m = self._m
        B = np.zeros((n, m))
        for j in range(m):
            du = np.zeros(m)
            step = eps * max(1.0, abs(u0[j]) if j < u0.size else 1.0)
            du[j] = step
            u_plus = u0.copy()
            u_plus[j] += step
            u_minus = u0.copy()
            u_minus[j] -= step
            f_plus = self.system.rhs(0.0, x_star, u_plus)
            f_minus = self.system.rhs(0.0, x_star, u_minus)
            B[:, j] = (f_plus - f_minus) / (2 * step)
        return B

    def design_local_gain(self, x_star: np.ndarray, u0: Optional[np.ndarray] = None) -> np.ndarray:
        """Solve the local CARE at manifold target x_star and return LQR gain K."""
        u0 = self.system.default_input() if u0 is None else u0
        A = self.system.jacobian(x_star, u0)
        B = self._numeric_B(x_star, u0)
        Q, R = self.params.Q, self.params.R
        try:
            P = solve_continuous_are(A, B, Q, R)
            K = np.linalg.solve(R, B.T @ P)
        except Exception:
            # Fall back to simple proportional damping if CARE is not solvable
            # at this operating point (e.g. B rank-deficient) -- keeps the
            # controller robust rather than raising mid-simulation.
            K = 0.5 * np.eye(self._m, self.system.n_states)
        return K

    # ------------------------------------------------------------------
    def precompute_gain_schedule(self, u_nominal: Optional[np.ndarray] = None) -> "ScheduledLQRControl":
        """
        Solve the local CARE once at *every* manifold sample point and
        cache the resulting gain matrices. This turns the runtime control
        law into an O(1) gain-schedule lookup (nearest manifold index ->
        cached K), which is what makes closed-loop time-domain simulation
        with an adaptive integrator tractable -- otherwise a fresh CARE
        solve triggered on every ODE function evaluation can dominate
        simulation cost by orders of magnitude.
        """
        u_nom = self.system.default_input() if u_nominal is None else np.asarray(u_nominal, dtype=float)
        self._gain_schedule = [self.design_local_gain(pt.x_star, u_nom) for pt in self.manifold.points]
        return self

    def as_controller(self, u_nominal: Optional[np.ndarray] = None) -> Callable[[float, np.ndarray], np.ndarray]:
        """
        Return a `controller(t, x) -> u` closure compatible with
        `core.simulator.Simulator.simulate(..., controller=...)`.

        Uses a precomputed gain schedule over the manifold (computed
        lazily on first call, or explicitly via `precompute_gain_schedule`)
        for O(1) runtime lookups.
        """
        u_nom = self.system.default_input() if u_nominal is None else np.asarray(u_nominal, dtype=float)
        if self._gain_schedule is None:
            self.precompute_gain_schedule(u_nom)

        def controller(t: float, x: np.ndarray) -> np.ndarray:
            self._call_count += 1
            idx, _ = self.manifold.nearest_index(x)
            x_star = self.manifold.points[idx].x_star
            K = self._gain_schedule[idx]

            u = u_nom - K @ (x - x_star)
            if self.params.u_min is not None:
                u = np.maximum(u, self.params.u_min)
            if self.params.u_max is not None:
                u = np.minimum(u, self.params.u_max)
            return u

        return controller


# Deprecated alias: retained for backward compatibility with case files,
# scripts, and documentation written against Version 1/2 of this platform.
# New code should use ScheduledLQRControl directly (see module docstring).
ManifoldReshapingControl = ScheduledLQRControl
