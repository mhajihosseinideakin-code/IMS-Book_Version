"""
control
-------

Per the architecture document (Part IV §4.6), this package maintains two
architecturally distinct controller libraries:

    Conventional Control Library   ->  control.mrc.ScheduledLQRControl
                                        (manifold-scheduled LQR; more
                                        conventional controllers such as
                                        PID/MPC/sliding-mode are future
                                        additions to this same library)

    IMS Control Framework          ->  control.mrc_synthesis.MRCSynthesizer
                                        (genuine Manifold-Reshaping Control:
                                        derives a closed-form nonlinear
                                        control law directly from a
                                        component's manifold constraint,
                                        per architecture document §1.7/§2.6)
"""

from .mrc import ScheduledLQRControl, ManifoldReshapingControl, MRCDesignParams
from .mrc_synthesis import MRCSynthesizer, MRCSynthesisResult, MRCSynthesisError

__all__ = [
    "ScheduledLQRControl", "ManifoldReshapingControl", "MRCDesignParams",
    "MRCSynthesizer", "MRCSynthesisResult", "MRCSynthesisError",
]
