"""
ims_platform.framework.interfaces
----------------------------------

The standardized contracts shared by every IMS analysis. Plain dataclasses so
they serialize cleanly to JSON at the API boundary and are trivial to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


#: The single, standard workflow every analysis is presented through.
STANDARD_WORKFLOW_STAGES: List[str] = [
    "Model/Network",
    "Analysis",
    "Disturbance",
    "Simulation",
    "IMS Results",
    "Report",
]

#: The four guarantee claim levels the platform keeps strictly distinct.
#: An analysis may only ever populate results at the level it actually
#: establishes; a lower level is never promoted to a stronger one.
CLAIM_LEVELS: Dict[str, str] = {
    "ideal_residual_contraction": (
        "EXACT (ideal unsaturated model): an algebraic closed-loop identity "
        "(e.g. de_Sigma/dt = -k_m e_Sigma) valid while the trajectory stays in the model domain."
    ),
    "local_equilibrium_stability": (
        "LOCAL ANALYTICAL: linearisation (Routh-Hurwitz / eigenvalues) shows local exponential "
        "stability of an isolated equilibrium. NOT a region of attraction."
    ),
    "observed_finite_horizon_recovery": (
        "NUMERICAL / EMPIRICAL: a single finite-horizon trajectory returning near the target is "
        "observed evidence only; sampled on the solver grid, not a continuous-time proof."
    ),
    "certified_regional_ims": (
        "CERTIFIED: an explicit forward-invariant region with a Lyapunov/contraction certificate. "
        "Established only by analyses that explicitly provide such a certificate."
    ),
}


@dataclass
class SupportedOutputs:
    """Which standard IMS outputs an analysis actually produces."""
    equilibrium: bool = False
    manifold_residual: bool = False
    residual_trajectory: bool = False
    contraction_evidence: bool = False
    local_stability: bool = False
    finite_horizon_recovery: bool = False
    solver_constraint_status: bool = False
    report_export: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return asdict(self)


@dataclass
class AnalysisDescriptor:
    """
    Metadata the Explorer analysis hub uses to list and route an analysis.

    `standardized=True` means the analysis is fully runnable through the shared
    framework endpoints (/api/analysis/{id}/run). `standardized=False` means it
    is a validated existing case whose bespoke, unchanged engine is reached via
    `entry` (so it is discoverable and connected to the common hub without a
    rewrite/regression).
    """
    id: str
    title: str
    model_id: str
    category: str                      # e.g. "controller", "dc_microgrid", "grid_forming", "network"
    description: str
    supported_outputs: SupportedOutputs
    workflow_stages: List[str] = field(default_factory=lambda: list(STANDARD_WORKFLOW_STAGES))
    guarantee_scope: str = ""
    standardized: bool = False
    entry: Dict[str, Any] = field(default_factory=dict)   # how the frontend opens it
    status: str = "available"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["supported_outputs"] = self.supported_outputs.to_dict()
        return d


@runtime_checkable
class Analysis(Protocol):
    """
    The interface every fully-standardized analysis implements. Existing
    validated cases that are not yet migrated only need a descriptor.
    """

    def descriptor(self) -> AnalysisDescriptor: ...

    def default_config(self) -> Dict[str, Any]: ...

    def validate(self, config: Dict[str, Any]) -> List[str]:
        """Return a list of admissibility errors ([] if the config is valid)."""
        ...

    def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute and return a standardized IMS result envelope (see build_envelope)."""
        ...


def build_envelope(
    *,
    analysis_id: str,
    model_id: str,
    model_version: str,
    supported: SupportedOutputs,
    equilibrium: Optional[Dict] = None,
    manifold: Optional[Dict] = None,
    residual_trajectory: Optional[Dict] = None,
    contraction_evidence: Optional[Dict] = None,
    local_stability: Optional[Dict] = None,
    recovery_status: Optional[Dict] = None,
    solver_status: Optional[Dict] = None,
    trajectories: Optional[Dict] = None,
    run_id: Optional[str] = None,
    provenance: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Assemble the standardized result envelope shared by all analyses."""
    return {
        "analysis_id": analysis_id,
        "model_id": model_id,
        "model_version": model_version,
        "workflow_stages": list(STANDARD_WORKFLOW_STAGES),
        "supported_outputs": supported.to_dict(),
        "equilibrium": equilibrium,
        "manifold": manifold,
        "residual_trajectory": residual_trajectory,
        "contraction_evidence": contraction_evidence,
        "local_stability": local_stability,
        "recovery_status": recovery_status,
        "solver_status": solver_status,
        "trajectories": trajectories,
        "claim_levels": CLAIM_LEVELS,
        "run_id": run_id,
        "provenance": provenance,
    }
