"""
IMS Platform
============

An open, modular, extensible engineering platform for nonlinear dynamic
modelling, large-signal stability assessment, and recoverability analysis
of converter-dominated power systems, built around the Intrinsic Manifold
Stability (IMS) framework and Manifold-Reshaping Control (MRC).

Design philosophy (mirrors the layered architecture of tools like
pandapower / MATPOWER, adapted for large-signal nonlinear analysis):

    ims_platform.core            -> generic nonlinear system + ODE simulator
    ims_platform.engine          -> Symbolic Engine (SymPy-based differentiation
                                     and compilation, architecture doc Part V)
    ims_platform.network         -> Network representation + Automatic Model
                                     Builder (Bus, Line, component library,
                                     architecture doc Part IV Section 4.1-4.4)
    ims_platform.models          -> concrete converter-dominated system models
                                     (hand-written; network.* is the automatic
                                     alternative for multi-bus topologies)
    ims_platform.ims             -> manifold identification, residuals,
                                     recoverability region / metrics
    ims_platform.control         -> two distinct controller libraries:
                                       - ScheduledLQRControl (Conventional
                                         Control Library: manifold-scheduled LQR)
                                       - MRCSynthesizer (IMS Control Framework:
                                         genuine, nonlinear Manifold-Reshaping
                                         Control, derived symbolically per
                                         architecture document §1.7/§2.6)
    ims_platform.visualization   -> plotting utilities
    ims_platform.reporting       -> automated Markdown/JSON engineering reports

Every layer works only through the public interfaces defined in
`core.system.DynamicalSystem`, so new device models, new manifold-
identification strategies, or new controllers can be added without
touching the rest of the codebase (open/closed principle).
"""

from .core.system import DynamicalSystem, EquilibriumResult
from .core.simulator import Simulator, TrajectoryResult
from .ims.manifold import IntrinsicManifold, ManifoldPoint
from .ims.recoverability import RecoverabilityAnalyzer, RecoverabilityReport
from .control.mrc import ScheduledLQRControl, ManifoldReshapingControl
from .control.mrc_synthesis import MRCSynthesizer, MRCSynthesisResult, MRCSynthesisError
from .network import Network, Bus, Line, AutomaticModelBuilder
from .network import ElectricalModel, Controller, Converter, ConverterModel
from .network import BuckModel, BoostModel, BuckBoostModel
from .network import SynthesizedMRCController, ConverterCPLElectricalModel

__version__ = "0.2.0"

__all__ = [
    "DynamicalSystem",
    "EquilibriumResult",
    "Simulator",
    "TrajectoryResult",
    "IntrinsicManifold",
    "ManifoldPoint",
    "RecoverabilityAnalyzer",
    "RecoverabilityReport",
    "ScheduledLQRControl",
    "ManifoldReshapingControl",
    "MRCSynthesizer",
    "MRCSynthesisResult",
    "MRCSynthesisError",
    "Network",
    "Bus",
    "Line",
    "AutomaticModelBuilder",
    "ElectricalModel",
    "Controller",
    "Converter",
    "ConverterModel",
    "BuckModel",
    "BoostModel",
    "BuckBoostModel",
    "SynthesizedMRCController",
    "ConverterCPLElectricalModel",
]
