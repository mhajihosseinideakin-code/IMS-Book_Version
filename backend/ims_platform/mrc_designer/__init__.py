"""
ims_platform.mrc_designer
--------------------------

General MRC Designer: the platform step that turns an assembled model + its
IMS analysis into a Manifold-Reshaping Control design, verifies it in closed
loop, and compares Before/After.

Scientific guardrails (enforced, not optional):
  * ONE source of truth: the Designer consumes the actual project model /
    manifold / equilibrium; it never substitutes the four-state reference for
    a user's network.
  * HYBRID authority: analytic (symbolic) G(x) and Dphi(x) are authoritative
    and enable analytic MRC establishment; finite-difference G(x) is used ONLY
    for numeric feasibility/diagnostics on assembled models and can NEVER, on
    its own, mark an MRC as analytically established.
  * NEVER auto-synthesize for an unsupported model, NEVER invent a target
    manifold, and NEVER promote numerical evidence to an analytical or
    certified-IMS claim. The four claim levels stay distinct.
  * The Four-State Stabilising MRC is the exact reference/regression case; the
    generic synthesis path must reproduce it within tolerance.
"""

from .feasibility import (
    control_affine_split,
    manifold_gradient,
    feasibility_report,
    SUPPORTED_ANALYTIC_MODELS,
)
from .designer import (
    synthesize_mrc,
    verify_closed_loop,
    before_after,
    MRCNotEstablished,
)

__all__ = [
    "control_affine_split",
    "manifold_gradient",
    "feasibility_report",
    "SUPPORTED_ANALYTIC_MODELS",
    "synthesize_mrc",
    "verify_closed_loop",
    "before_after",
    "MRCNotEstablished",
]
