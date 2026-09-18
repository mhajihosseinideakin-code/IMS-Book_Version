"""
network.converter_topologies
------------------------------

Concrete averaged (continuous-conduction-mode, CCM) DC-DC converter
topologies, built on `network.electrical_model.ConverterModel`
(architecture document Part IV Section 4.4, extended per the project's
own V2.4 refinement to introduce a shared base beneath ElectricalModel
for the whole converter family).

Every model here shares the same local state -- the inductor current
i_L -- and the same input-rail parameter v_in (see ConverterModel's
docstring for why the input side is a fixed parameter rather than a
second network bus in this version). Each subclass differs only in its
averaged inductor-voltage and port-current equations, which are the
standard textbook CCM averaged relations for that topology.

Validation strategy
--------------------
Each topology has a well-known, textbook ideal (R_L = 0) steady-state
voltage conversion ratio:

    Buck:       V_o / V_in = D
    Boost:      V_o / V_in = 1 / (1 - D)
    Buck-Boost: V_o / V_in = D / (1 - D)   (output-magnitude convention;
                                             see BuckBoostModel docstring)

`tests/test_converter_topologies.py` checks the equilibrium found by the
platform's own (topology-agnostic) Newton solver against these formulas
directly -- the same "check against an independently-known-correct
result" discipline used to validate MRC synthesis and the network
assembler.
"""

from __future__ import annotations

import numpy as np
from typing import Dict

from .electrical_model import ConverterModel


class BuckModel(ConverterModel):
    """
    Averaged buck (step-down) converter. Inductor current i_L is the
    converter's own state; it is also, by the standard buck topology,
    exactly the current delivered to the output bus.

        L * di_L/dt = d*v_in - v_bus - R_L*i_L
        port_current = i_L

    Ideal (R_L=0) steady-state ratio: V_o/V_in = D.
    """

    state_names = ("i_L",)
    param_names = ("v_in", "L", "R_L")

    def port_current(self, v_bus: float, local_x: np.ndarray, control_signal: float, p: Dict) -> float:
        i_L = local_x[0]
        return i_L

    def local_dynamics(self, v_bus: float, local_x: np.ndarray, control_signal: float, p: Dict) -> np.ndarray:
        i_L = local_x[0]
        d = self.clamp_duty(control_signal)
        di_L = (d * p["v_in"] - v_bus - p["R_L"] * i_L) / p["L"]
        return np.array([di_L])


class BoostModel(ConverterModel):
    """
    Averaged boost (step-up) converter. Inductor current i_L sits on the
    input side; only a (1-d) fraction of it reaches the output on
    average (the complementary fraction recirculates through the switch).

        L * di_L/dt = v_in - (1-d)*v_bus - R_L*i_L
        port_current = (1-d)*i_L

    Ideal (R_L=0) steady-state ratio: V_o/V_in = 1/(1-D).
    """

    state_names = ("i_L",)
    param_names = ("v_in", "L", "R_L")

    def port_current(self, v_bus: float, local_x: np.ndarray, control_signal: float, p: Dict) -> float:
        i_L = local_x[0]
        d = self.clamp_duty(control_signal)
        return (1.0 - d) * i_L

    def local_dynamics(self, v_bus: float, local_x: np.ndarray, control_signal: float, p: Dict) -> np.ndarray:
        i_L = local_x[0]
        d = self.clamp_duty(control_signal)
        di_L = (p["v_in"] - (1.0 - d) * v_bus - p["R_L"] * i_L) / p["L"]
        return np.array([di_L])


class BuckBoostModel(ConverterModel):
    """
    Averaged (inverting) buck-boost converter, in the standard
    output-magnitude convention: the physical output polarity is
    inverted relative to the input, but this model tracks the output
    bus voltage as a positive magnitude v_bus (matching every other
    component in this platform, which assumes v_bus > 0 -- see
    `network.bus.Bus.v_min`), with the sign inversion absorbed into how
    the averaged equations are written rather than into v_bus itself.

        L * di_L/dt = d*v_in - (1-d)*v_bus - R_L*i_L
        port_current = (1-d)*i_L

    Ideal (R_L=0) steady-state ratio: V_o/V_in = D/(1-D).
    """

    state_names = ("i_L",)
    param_names = ("v_in", "L", "R_L")

    def port_current(self, v_bus: float, local_x: np.ndarray, control_signal: float, p: Dict) -> float:
        i_L = local_x[0]
        d = self.clamp_duty(control_signal)
        return (1.0 - d) * i_L

    def local_dynamics(self, v_bus: float, local_x: np.ndarray, control_signal: float, p: Dict) -> np.ndarray:
        i_L = local_x[0]
        d = self.clamp_duty(control_signal)
        di_L = (d * p["v_in"] - (1.0 - d) * v_bus - p["R_L"] * i_L) / p["L"]
        return np.array([di_L])
