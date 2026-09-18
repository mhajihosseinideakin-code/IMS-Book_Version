"""
ims_platform.framework.registry
--------------------------------

The single registry the Explorer analysis hub reads. It holds:

  * fully-standardized analyses (implement the `Analysis` interface and are
    runnable through /api/analysis/{id}/run) -- currently the Four-State
    Stabilising MRC; and

  * the existing validated Case Library studies, registered as descriptors
    that declare which standard IMS outputs they support and point (`entry`)
    at their unchanged, validated engines/views. This connects them to the
    common hub and standard workflow metadata WITHOUT rewriting or regressing
    them; migrating each to the runnable interface is future, incremental work.

Adding a new analysis = add one entry here. No new application per case.
"""

from __future__ import annotations

from typing import Dict, List

from .interfaces import AnalysisDescriptor, SupportedOutputs, Analysis
from .analyses import StabilizingMRCAnalysis


# --- Fully-standardized, runnable analyses -------------------------------
_MRC = StabilizingMRCAnalysis()

ANALYSIS_REGISTRY: Dict[str, Analysis] = {
    _MRC.analysis_id: _MRC,
}


# --- Existing validated Case Library studies (discoverable, preserved) ---
# supported_outputs reflect what each existing engine genuinely provides today;
# `standardized=False` means "runs via its existing validated view/engine".
_CASE_DESCRIPTORS: List[AnalysisDescriptor] = [
    AnalysisDescriptor(
        id="iberian_2025_overvoltage_cascade",
        title="Iberian 2025-Inspired Overvoltage Cascade",
        model_id="iberian_scenario",
        category="network",
        description="Converter-network-protection interaction: insufficient dynamic voltage support, "
                    "current limiting, and delayed overvoltage protection cascading through a weak grid.",
        supported_outputs=SupportedOutputs(
            equilibrium=True, local_stability=True, finite_horizon_recovery=True,
            solver_constraint_status=True, report_export=True,
        ),
        guarantee_scope="Scenario stress-test + IMS geometry; observed finite-horizon behaviour, not a certified basin.",
        standardized=False,
        entry={"type": "legacy_view", "view_fn": "showIberianScenario", "api_prefix": "/api/iberian"},
    ),
    AnalysisDescriptor(
        id="dc_microgrid_cpl",
        title="Converter-CPL Voltage Collapse",
        model_id="dc_microgrid_cpl",
        category="dc_microgrid",
        description="Classical Middlebrook large-signal instability in a DC microgrid feeding a constant-power "
                    "load -- a genuine stable/saddle equilibrium pair.",
        supported_outputs=SupportedOutputs(
            equilibrium=True, manifold_residual=True, local_stability=True,
            finite_horizon_recovery=True, solver_constraint_status=True, report_export=True,
        ),
        guarantee_scope="Bistable equilibrium structure; manifold + recoverability via the generic model engine.",
        standardized=False,
        entry={"type": "legacy_project", "view_fn": "openProject", "project_id": "dc_microgrid_cpl"},
    ),
    AnalysisDescriptor(
        id="grid_forming_inverter",
        title="Weak-Grid GFL Synchronization",
        model_id="grid_forming_inverter",
        category="grid_forming",
        description="Droop-controlled single converter on a weak grid connection; swing-type large-signal dynamics.",
        supported_outputs=SupportedOutputs(
            equilibrium=True, manifold_residual=True, local_stability=True,
            finite_horizon_recovery=True, solver_constraint_status=True, report_export=True,
        ),
        guarantee_scope="Swing-type equilibrium + manifold/recoverability via the generic model engine.",
        standardized=False,
        entry={"type": "legacy_project", "view_fn": "openProject", "project_id": "grid_forming_inverter"},
    ),
    AnalysisDescriptor(
        id="gfm_current_limit_recovery",
        title="GFM Current-Limit Recovery",
        model_id="gfm_current_limit",
        category="grid_forming",
        description="A grid-forming converter's fault-ride-through: how current headroom sets the voltage-dip "
                    "depth during a nearby fault and whether it stays within an admissible ride-through envelope.",
        supported_outputs=SupportedOutputs(
            finite_horizon_recovery=True, solver_constraint_status=True, report_export=True,
        ),
        guarantee_scope="Fault-ride-through envelope (constraint status); observed recovery, not a certified basin.",
        standardized=False,
        entry={"type": "legacy_view", "view_fn": "showGfmCurrentLimitCase", "api_prefix": "/api/gfm_current_limit"},
    ),
    AnalysisDescriptor(
        id="multi_converter_fault_recovery",
        title="Multi-Converter Fault Recovery",
        model_id="multi_converter_fault",
        category="network",
        description="Two interacting converters (grid-forming anchor + grid-following follower) sharing a network "
                    "and a common local fault -- local vs. remote support and combined fault-ride-through.",
        supported_outputs=SupportedOutputs(
            finite_horizon_recovery=True, solver_constraint_status=True, report_export=True,
        ),
        guarantee_scope="Interacting-converter fault-ride-through; observed recovery, not a certified basin.",
        standardized=False,
        entry={"type": "legacy_view", "view_fn": "showMultiConverterCase", "api_prefix": "/api/multi_converter_fault"},
    ),
]

_CASE_DESCRIPTOR_INDEX: Dict[str, AnalysisDescriptor] = {d.id: d for d in _CASE_DESCRIPTORS}


def get_analysis(analysis_id: str) -> Analysis:
    """Return a runnable Analysis, or raise KeyError if not standardized/known."""
    return ANALYSIS_REGISTRY[analysis_id]


def list_descriptors() -> List[dict]:
    """
    Every registered study for the Explorer analysis hub: runnable analyses
    first, then the connected validated cases. Ordered, JSON-ready.
    """
    out: List[dict] = []
    for analysis in ANALYSIS_REGISTRY.values():
        out.append(analysis.descriptor().to_dict())
    out.extend(d.to_dict() for d in _CASE_DESCRIPTORS)
    return out


def get_descriptor(analysis_id: str) -> dict:
    if analysis_id in ANALYSIS_REGISTRY:
        return ANALYSIS_REGISTRY[analysis_id].descriptor().to_dict()
    if analysis_id in _CASE_DESCRIPTOR_INDEX:
        return _CASE_DESCRIPTOR_INDEX[analysis_id].to_dict()
    raise KeyError(analysis_id)
