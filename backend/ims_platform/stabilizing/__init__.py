"""
ims_platform.stabilizing
-------------------------

Verified end-to-end workflow layer for the IDEAL continuous-time
four-state stabilizing-MRC model x = (i_l, v_b, sigma, v_o).

This package does NOT re-derive any mathematics. It is a thin
orchestration layer on top of the already-verified
`ims_platform.models.stabilizing_mrc.StabilizingMRCModel` (whose control
law, equilibrium, characteristic coefficients and Routh-Hurwitz report
are checked in tests/test_stabilizing_mrc.py) and the existing
`ims_platform.core.simulator.Simulator`.

It adds only what the four-state milestone requires and the core did not
yet expose:
  - explicit parameter admissibility validation (errors, not silent runs);
  - numerical closed-loop equilibrium residual ||f_cl(x*)||;
  - full 4x4 closed-loop Jacobian (analytic) cross-checked against an
    INDEPENDENT finite-difference Jacobian of the closed-loop field;
  - transverse-mode verification lambda_perp == -k_m;
  - closed-loop simulation with a temporary CPL power disturbance applied
    as a time-dependent parameter change (not a state hack);
  - off-manifold residual-contraction comparison against e0*exp(-k_m t);
  - single-run recovery status vocabulary (section 8 semantics);
  - one verified dataset reused for UI plots, CSV export and PDF report.
"""

from .model import StabilizingMRCClosedLoop
from .workflow import (
    MODEL_ID,
    MODEL_VERSION,
    validate_params,
    ParameterError,
    compute_equilibrium,
    compute_stability,
    run_simulation,
    dataset_to_csv,
)

__all__ = [
    "StabilizingMRCClosedLoop",
    "MODEL_ID",
    "MODEL_VERSION",
    "validate_params",
    "ParameterError",
    "compute_equilibrium",
    "compute_stability",
    "run_simulation",
    "dataset_to_csv",
]
