"""
visualization.plots
--------------------

Matplotlib-based visualisation utilities mirroring the IMS Platform's
intended dashboard views: manifold + trajectory overlay, residual time
series, and the recoverability map / index.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Dict

from ..ims.manifold import IntrinsicManifold
from ..core.simulator import TrajectoryResult
from ..ims.recoverability import RecoverabilityReport


def plot_manifold_and_trajectory(
    manifold: IntrinsicManifold,
    trajectory: Optional[TrajectoryResult] = None,
    state_idx=(0, 1),
    labels: Optional[Sequence[str]] = None,
    title: str = "Intrinsic Manifold and System Trajectory",
    ax=None,
):
    """2-D projection of the intrinsic manifold branch with an optional overlaid trajectory."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 5.5))

    S = manifold.samples
    stable_mask = manifold.stable_branch_mask()
    i, j = state_idx

    ax.plot(S[stable_mask, i], S[stable_mask, j], color="#2e7d32", lw=2.5, label="Intrinsic manifold M (stable branch)")
    if np.any(~stable_mask):
        ax.plot(S[~stable_mask, i], S[~stable_mask, j], color="#c62828", lw=2, ls="--", label="M (unstable branch)")

    if trajectory is not None:
        ax.plot(trajectory.x[i, :], trajectory.x[j, :], color="#1565c0", lw=1.8, label="System trajectory x(t)")
        ax.scatter([trajectory.x[i, 0]], [trajectory.x[j, 0]], color="orange", zorder=5, s=60, label="Disturbed state")
        ax.scatter([trajectory.x[i, -1]], [trajectory.x[j, -1]], color="black", marker="*", zorder=5, s=140, label="Final state")

    labels = labels or (f"x[{i}]", f"x[{j}]")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    if own_fig:
        fig.tight_layout()
        return fig
    return ax


def plot_residual_timeseries(trajectory: TrajectoryResult, manifold: IntrinsicManifold, ax=None, title: str = "Manifold Residual r_m(t)"):
    """Time evolution of the manifold residual along a trajectory."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 3.5))

    r = manifold.residual_trajectory(trajectory.x)
    ax.plot(trajectory.t, r, color="#8e24aa", lw=2)
    ax.fill_between(trajectory.t, 0, r, color="#8e24aa", alpha=0.15)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("r_m")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    if own_fig:
        fig.tight_layout()
        return fig
    return ax


def plot_recoverability_map(report: RecoverabilityReport, state_idx=(0, 1), labels=None, ax=None, title: str = "Recoverability Map"):
    """Scatter of sampled disturbed states classified recoverable / non-recoverable."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))

    i, j = state_idx
    rec = report.labels
    ax.scatter(report.samples[rec, i], report.samples[rec, j], color="#2e7d32", s=18, alpha=0.75, label=f"Recoverable ({report.n_recoverable})")
    ax.scatter(report.samples[~rec, i], report.samples[~rec, j], color="#c62828", s=18, alpha=0.75, label=f"Non-recoverable ({report.n_samples - report.n_recoverable})")
    ax.scatter([report.nominal_state[i]], [report.nominal_state[j]], color="black", marker="*", s=160, zorder=5, label="Nominal state")

    labels = labels or (f"x[{i}]", f"x[{j}]")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(f"{title}  (Recoverability Index = {report.recoverability_index:.2f}, risk = {report.risk_level})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    if own_fig:
        fig.tight_layout()
        return fig
    return ax


def plot_recoverability_curve(reports_by_radius: Dict[float, RecoverabilityReport], ax=None, title: str = "Recoverability Index vs. Disturbance Radius"):
    """Recoverability index as a function of disturbance magnitude -- a compact resilience summary curve."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6.5, 4))

    radii = sorted(reports_by_radius.keys())
    idx = [reports_by_radius[r].recoverability_index for r in radii]
    ax.plot(radii, idx, marker="o", color="#00695c", lw=2)
    ax.axhline(0.85, color="green", ls=":", lw=1, alpha=0.6, label="LOW risk threshold")
    ax.axhline(0.6, color="orange", ls=":", lw=1, alpha=0.6, label="MEDIUM risk threshold")
    ax.set_xlabel("Disturbance radius")
    ax.set_ylabel("Recoverability Index")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    if own_fig:
        fig.tight_layout()
        return fig
    return ax
