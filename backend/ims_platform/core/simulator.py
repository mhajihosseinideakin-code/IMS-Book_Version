"""
core.simulator
--------------

Nonlinear time-domain simulation engine. Wraps `scipy.integrate.solve_ivp`
with a uniform interface used across the platform (IMS residual evaluation,
recoverability sampling, MRC closed-loop validation), and supports:

    - disturbance events (parameter or state jumps at a given time)
    - closed-loop simulation with an arbitrary feedback controller
    - dense output for smooth trajectory post-processing
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
from scipy.integrate import solve_ivp

from .system import DynamicalSystem


@dataclass
class TrajectoryResult:
    """Container for a simulated trajectory."""
    t: np.ndarray
    x: np.ndarray  # shape (n_states, n_time)
    events: List[Tuple[float, str]] = field(default_factory=list)
    success: bool = True
    message: str = ""

    def state_at(self, t_query: float) -> np.ndarray:
        idx = int(np.argmin(np.abs(self.t - t_query)))
        return self.x[:, idx]

    @property
    def final_state(self) -> np.ndarray:
        return self.x[:, -1]


class Simulator:
    """
    Generic nonlinear time-domain simulator for any `DynamicalSystem`.

    Parameters
    ----------
    system : DynamicalSystem
        The model to simulate.
    method : str
        Integrator passed to scipy.integrate.solve_ivp (default 'RK45';
        use 'Radau' or 'BDF' for stiff converter-switching-averaged models).
    rtol, atol : float
        Integration tolerances.
    """

    def __init__(self, system: DynamicalSystem, method: str = "RK45", rtol: float = 1e-6, atol: float = 1e-8):
        self.system = system
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def simulate(
        self,
        x0: np.ndarray,
        t_span: Tuple[float, float],
        u: Optional[np.ndarray] = None,
        controller: Optional[Callable[[float, np.ndarray], np.ndarray]] = None,
        disturbances: Optional[List[Dict]] = None,
        n_eval: int = 400,
    ) -> TrajectoryResult:
        """
        Simulate the (possibly disturbed, possibly closed-loop) system.

        controller(t, x) -> u   overrides `u` if provided (closed-loop mode).
        disturbances: list of {"t": time, "type": "state_jump"|"param", "value": ...}
            applied by segmenting the integration at each disturbance time.
        """
        u0 = self.system.default_input() if u is None else np.asarray(u, dtype=float)
        disturbances = sorted(disturbances or [], key=lambda d: d["t"])

        segment_bounds = [t_span[0]] + [d["t"] for d in disturbances if t_span[0] < d["t"] < t_span[1]] + [t_span[1]]
        segment_bounds = sorted(set(segment_bounds))

        t_all: List[np.ndarray] = []
        x_all: List[np.ndarray] = []
        events: List[Tuple[float, str]] = []
        x_current = np.asarray(x0, dtype=float)
        base_params = dict(self.system.params)

        for i in range(len(segment_bounds) - 1):
            t0, t1 = segment_bounds[i], segment_bounds[i + 1]

            # apply any disturbance scheduled exactly at t0 (after the first segment)
            for d in disturbances:
                if np.isclose(d["t"], t0) and i > 0:
                    if d["type"] == "state_jump":
                        x_current = x_current + np.asarray(d["value"], dtype=float)
                        events.append((t0, f"state_jump {d.get('label', '')}"))
                    elif d["type"] == "param":
                        self.system.params.update(d["value"])
                        events.append((t0, f"param_change {d.get('label', '')}"))

            def rhs(t, x, _u0=u0, _controller=controller):
                u_eff = _controller(t, x) if _controller is not None else _u0
                return self.system.rhs(t, x, u_eff)

            n_pts = max(20, int(n_eval * (t1 - t0) / max(t_span[1] - t_span[0], 1e-9)))
            t_eval = np.linspace(t0, t1, n_pts)

            sol = solve_ivp(
                rhs, (t0, t1), x_current, method=self.method,
                t_eval=t_eval, rtol=self.rtol, atol=self.atol, dense_output=False,
            )

            t_all.append(sol.t)
            x_all.append(sol.y)
            x_current = sol.y[:, -1]

            if not sol.success:
                self.system.params = base_params
                return TrajectoryResult(
                    t=np.concatenate(t_all), x=np.concatenate(x_all, axis=1),
                    events=events, success=False, message=sol.message,
                )

        self.system.params = base_params  # restore nominal params after disturbance segments
        t_full = np.concatenate(t_all)
        x_full = np.concatenate(x_all, axis=1)
        return TrajectoryResult(t=t_full, x=x_full, events=events, success=True, message="ok")
