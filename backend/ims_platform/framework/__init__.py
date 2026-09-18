"""
ims_platform.framework
-----------------------

Shared, extensible IMS analysis framework.

Goal (Phase-1 completion): turn the platform's collection of capabilities
into ONE general analysis engine so future models / controllers / networks /
recoverability studies can be added by implementing a single interface,
instead of building a separate application per case.

Nothing here re-derives mathematics or rewrites an existing validated case.
It standardizes:
  * the WORKFLOW every analysis follows
    (Project -> Model/Network -> Analysis -> Disturbance -> Simulation ->
     IMS Results -> Report);
  * the ANALYSIS interface (descriptor / default_config / validate / run);
  * the RESULT envelope carrying the core IMS outputs where supported
    (equilibrium, manifold residual, residual trajectory, contraction
     evidence, local stability, finite-horizon recovery, solver/constraint
     status) with the analytical-vs-numerical claim distinction preserved.

Existing validated cases (Iberian, GFM current-limit, multi-converter fault,
DC-microgrid CPL, weak-grid GFL) are registered as first-class analyses in
the same registry, declaring which standard outputs they support and where
their existing, unchanged engine lives. The Four-State Stabilising-MRC is the
first analysis fully conformant to the standardized workflow.
"""

from .interfaces import (
    STANDARD_WORKFLOW_STAGES,
    SupportedOutputs,
    AnalysisDescriptor,
    Analysis,
    CLAIM_LEVELS,
)
from .registry import list_descriptors, get_analysis, ANALYSIS_REGISTRY

__all__ = [
    "STANDARD_WORKFLOW_STAGES",
    "SupportedOutputs",
    "AnalysisDescriptor",
    "Analysis",
    "CLAIM_LEVELS",
    "list_descriptors",
    "get_analysis",
    "ANALYSIS_REGISTRY",
]
