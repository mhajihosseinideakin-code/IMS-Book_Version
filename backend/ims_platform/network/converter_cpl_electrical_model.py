"""
network.converter_cpl_electrical_model
------------------------------------------

`ConverterCPLElectricalModel`: the network-layer decomposition of the
confidential manuscript's converter--CPL system
(`models.converter_cpl_paper.ConverterCPLPaper`), used to prove that
Manifold-Reshaping Control derived by `control.mrc_synthesis.MRCSynthesizer`
against that standalone 3-state model can drive an actual Converter
inside an assembled Network -- finalizing the controller interface
(Phase V2.5 preparation) by connecting the platform's two previously
disconnected halves: IMS-native control synthesis and the Network/
Converter object model.

The decomposition
-------------------
The paper's system is

    L * i_l_dot = v_o - R*i_l - v_b
    C * v_b_dot = i_l - P/v_b
    v_o_dot     = u

Only the first and third equations are this converter's own physics; the
second is exactly what a `network.bus.Bus` (capacitance C, voltage
state v_b) already does automatically once a `ConstantPowerLoad(P)` is
attached to it (Phase V2.2) -- so the network-layer version of this
converter needs no bus-side equation of its own at all. Its local state
is (i_l, v_o), its port current is i_l (the current the paper's own bus
equation expects), and its local dynamics are exactly the first and
third equations above, with v_bus taking the place of the paper's v_b.

Composing Bus(C) + ConstantPowerLoad(P) + Converter(this ElectricalModel)
therefore reproduces the paper's system exactly -- verified numerically
in `tests/test_controller_interface.py`.
"""

from __future__ import annotations

import numpy as np
from typing import Dict

from .electrical_model import ElectricalModel


class ConverterCPLElectricalModel(ElectricalModel):
    """
    Network-layer physics for the confidential manuscript's converter,
    excluding the bus-side equation (owned by the attached `Bus` +
    `ConstantPowerLoad` instead -- see module docstring).
    """

    state_names = ("i_l", "v_o")
    param_names = ("R", "L")
    control_mode = "grid_forming"

    def port_current(self, v_bus: float, local_x: np.ndarray, control_signal: float, p: Dict) -> float:
        i_l, v_o = local_x
        return i_l

    def local_dynamics(self, v_bus: float, local_x: np.ndarray, control_signal: float, p: Dict) -> np.ndarray:
        i_l, v_o = local_x
        di_l = (v_o - p["R"] * i_l - v_bus) / p["L"]
        dv_o = control_signal  # u, supplied by the paired Controller
        return np.array([di_l, dv_o])
