"""
ims.recoverability
-------------------

Implements the IMS notions of **Recoverability Region** R, **Critical
Boundary** dR, and quantitative **Recoverability Metrics**.

Methodology
-----------
Given the intrinsic manifold M (see `ims.manifold.IntrinsicManifold`),
we estimate R by Monte-Carlo sampling of disturbed initial conditions in
a neighbourhood of M, forward-simulating each one with the (open- or
closed-loop) system, and classifying the outcome:

    - RECOVERABLE      : trajectory stays admissible and its terminal
                          manifold residual falls below `recovery_tol`
    - NON-RECOVERABLE  : trajectory leaves the admissible operating
                          region, diverges, or fails to reconverge to M
                          within the simulation horizon

The classified sample cloud gives:
    - Recoverability Index      = (#recoverable) / (#sampled)
    - Margin to (critical) boundary = smallest residual among
                                       non-recoverable samples (a lower
                                       bound on distance from the nominal
                                       point to dR)
    - A discrete projection of dR onto state space (the classified
      sample cloud itself, or a coarse grid-based boundary estimate)

This Monte-Carlo formulation is intentionally solver/model-agnostic: it
works for any `DynamicalSystem`, is trivially parallelisable, and can be
swapped for more advanced techniques (e.g. level-set/HJB methods,
sum-of-squares certificates) without changing the rest of the platform.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ..core.system import DynamicalSystem
from ..core.simulator import Simulator
from .manifold import IntrinsicManifold


@dataclass
class RecoverabilityReport:
    """Result bundle from a recoverability assessment sweep."""
    samples: np.ndarray                 # (n_samples, n_states) disturbed initial states
    labels: np.ndarray                  # (n_samples,) bool: True = recoverable
    terminal_residual: np.ndarray       # (n_samples,) manifold residual at t_final
    recoverability_index: float
    margin_to_boundary: float
    n_recoverable: int
    n_samples: int
    nominal_state: np.ndarray
    risk_level: str = field(init=False)

    def __post_init__(self):
        if self.recoverability_index >= 0.85:
            self.risk_level = "LOW"
        elif self.recoverability_index >= 0.6:
            self.risk_level = "MEDIUM"
        else:
            self.risk_level = "HIGH"

    def summary(self) -> str:
        return (
            f"Recoverability Index : {self.recoverability_index:.2f}\n"
            f"Recoverable scenarios: {self.n_recoverable}/{self.n_samples} "
            f"({100*self.recoverability_index:.1f}%)\n"
            f"Margin to boundary   : {self.margin_to_boundary:.4f}\n"
            f"Risk level           : {self.risk_level}"
        )


class RecoverabilityAnalyzer:
    """
    Monte-Carlo recoverability assessment for a `DynamicalSystem`
    relative to a previously built `IntrinsicManifold`.
    """

    def __init__(
        self,
        system: DynamicalSystem,
        manifold: IntrinsicManifold,
        simulator: Optional[Simulator] = None,
        recovery_tol: float = 5e-2,
    ):
        self.system = system
        self.manifold = manifold
        self.simulator = simulator or Simulator(system)
        self.recovery_tol = recovery_tol

    # ------------------------------------------------------------------
    def sample_disturbances(
        self,
        nominal_state: np.ndarray,
        radius: float,
        n_samples: int = 200,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Uniformly sample disturbed initial states on a ball of given radius around nominal_state."""
        rng = rng or np.random.default_rng(0)
        n = self.system.n_states
        directions = rng.normal(size=(n_samples, n))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
        radii = radius * rng.random(n_samples) ** (1.0 / n)  # uniform-in-ball
        return nominal_state[None, :] + directions * radii[:, None]

    def assess(
        self,
        nominal_state: np.ndarray,
        radius: float,
        t_horizon: float = 20.0,
        n_samples: int = 200,
        u: Optional[np.ndarray] = None,
        controller: Optional[Callable[[float, np.ndarray], np.ndarray]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> RecoverabilityReport:
        """
        Estimate the recoverability region within `radius` of
        `nominal_state` by simulating `n_samples` disturbed trajectories
        for `t_horizon` seconds and classifying each against the
        intrinsic manifold.
        """
        disturbed = self.sample_disturbances(nominal_state, radius, n_samples, rng)
        labels = np.zeros(n_samples, dtype=bool)
        terminal_residual = np.full(n_samples, np.nan)

        for i in range(n_samples):
            traj = self.simulator.simulate(
                disturbed[i], (0.0, t_horizon), u=u, controller=controller, n_eval=150,
            )
            if not traj.success:
                terminal_residual[i] = np.inf
                labels[i] = False
                continue

            x_final = traj.final_state
            admissible_path = all(self.system.admissible(traj.x[:, k]) for k in range(0, traj.x.shape[1], 5))
            r_final = self.manifold.residual(x_final)
            terminal_residual[i] = r_final
            labels[i] = bool(admissible_path and r_final < self.recovery_tol and np.all(np.isfinite(x_final)))

        n_recoverable = int(np.sum(labels))
        recoverability_index = n_recoverable / n_samples if n_samples else 0.0

        non_rec_mask = ~labels
        if np.any(non_rec_mask):
            dist_to_nominal = np.linalg.norm(disturbed[non_rec_mask] - nominal_state[None, :], axis=1)
            margin_to_boundary = float(np.min(dist_to_nominal))
        else:
            margin_to_boundary = float(radius)  # no failures observed within sampled ball

        return RecoverabilityReport(
            samples=disturbed,
            labels=labels,
            terminal_residual=terminal_residual,
            recoverability_index=recoverability_index,
            margin_to_boundary=margin_to_boundary,
            n_recoverable=n_recoverable,
            n_samples=n_samples,
            nominal_state=np.asarray(nominal_state, dtype=float),
        )

    # ------------------------------------------------------------------
    def sweep_radius(
        self,
        nominal_state: np.ndarray,
        radii: List[float],
        t_horizon: float = 20.0,
        n_samples: int = 60,
        u: Optional[np.ndarray] = None,
        controller: Optional[Callable[[float, np.ndarray], np.ndarray]] = None,
    ) -> Dict[float, RecoverabilityReport]:
        """Assess recoverability index as a function of disturbance radius (a 'recoverability curve')."""
        rng = np.random.default_rng(42)
        return {
            r: self.assess(nominal_state, r, t_horizon=t_horizon, n_samples=n_samples,
                            u=u, controller=controller, rng=rng)
            for r in radii
        }
