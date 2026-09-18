"""
case.registry
--------------

Registry of system models the case-based interface can build. This is the
single place a new model needs to be listed to become available through
the YAML/JSON case files, the interactive wizard, and the CLI -- everything
else in `case/` works generically against `DynamicalSystem`.
"""

from __future__ import annotations

from typing import Dict, Type

from ..core.system import DynamicalSystem
from ..models import GridFormingInverter, DCMicrogridCPL, ConverterCPLPaper, StabilizingMRCModel


MODEL_REGISTRY: Dict[str, Type[DynamicalSystem]] = {
    "grid_forming_inverter": GridFormingInverter,
    "dc_microgrid_cpl": DCMicrogridCPL,
    "electrical_constraint_mrc": ConverterCPLPaper,
    "stabilizing_mrc": StabilizingMRCModel,
}

#: Human-facing metadata used by the wizard and by validation error messages.
#: `param_help` documents every entry accepted in a case file's `params:` block.
MODEL_METADATA: Dict[str, dict] = {
    "grid_forming_inverter": {
        "label": "Droop-controlled grid-forming inverter (single converter, swing-type dynamics)",
        "states": ["delta (power angle, rad)", "omega (frequency deviation, rad/s)"],
        "inputs": ["P_set (active-power setpoint, p.u.)"],
        "param_help": {
            "E": "Inverter internal EMF magnitude (p.u.). Typical: 0.9-1.1",
            "V": "Grid bus voltage magnitude (p.u.). Typical: 0.9-1.1",
            "X": "Line reactance between inverter and grid (p.u.). Typical: 0.1-0.5",
            "tau_p": "Active-power droop filter time constant (s). Typical: 0.01-0.2",
            "m_p": "Droop gain (p.u. power / p.u. frequency error). Typical: 0.5-5",
        },
        "default_operating_input": [0.5],
        "default_state_guess": [0.3, 0.0],
        "default_manifold_range": [0.0, 0.95],
    },
    "dc_microgrid_cpl": {
        "label": "DC microgrid bus feeding a constant-power load (bistable large-signal case)",
        "states": ["i (inductor/line current, p.u.)", "v (DC bus voltage, p.u.)"],
        "inputs": ["P_load (constant-power load demand, p.u.)"],
        "param_help": {
            "L": "Line/converter inductance (H, normalised). Typical: 0.001-0.02",
            "C": "DC bus capacitance (F, normalised). Typical: 0.01-0.1",
            "r": "Line resistance / damping (p.u.). Must be large enough relative to "
                 "P/(C v^2) at operating power for a stable high-voltage equilibrium "
                 "to exist (Middlebrook criterion). Typical: 0.05-0.3",
            "Vin": "Source-side regulated voltage (p.u.). Typical: 1.0-1.3",
            "v_floor": "Numerical smoothing floor for the CPL's 1/v term (p.u.). Leave at default unless "
                       "you understand the smooth-clamp implementation.",
        },
        "default_operating_input": [0.5],
        "default_state_guess": [0.44, 1.13],
        "default_manifold_range": [0.05, 1.2],
    },
    "electrical_constraint_mrc": {
        "label": "Electrical-constraint MRC (DIAGNOSTIC REFERENCE ONLY -- residual decay does NOT "
                 "stabilize an isolated equilibrium; see models.converter_cpl_paper docstring)",
        "states": ["i_l (inductor/line current, A)", "v_b (DC bus voltage, V)", "v_o (converter terminal voltage, V)"],
        "inputs": ["u (outer-loop actuation, u = dv_o/dt, V/s)"],
        "param_help": {
            "R": "Line resistance (Ohm). Typical: 0.1-0.5",
            "L": "Line inductance (H). Typical: 1e-3 to 5e-3",
            "C": "Bus capacitance (F). Typical: 1e-3 to 5e-3",
            "P": "Constant-power load (W). Typical: 5e3-3e4",
        },
        "default_operating_input": [0.0],
        "default_state_guess": [50.0, 400.0, 410.0],
        "default_manifold_range": [40.0, 500.0],
        "guarantee_scope": "This model's closed-loop equilibrium has a ZERO eigenvalue and a "
                            "structurally POSITIVE tangential mode (P/(C*v_b_star^2)) -- it demonstrates "
                            "why exact residual decay alone is insufficient for stabilization, and is "
                            "not itself a validated stabilizing controller.",
    },
    "stabilizing_mrc": {
        "label": "Stabilizing MRC (book construction: internal integrator target manifold with an "
                 "isolated, locally exponentially stable equilibrium)",
        "states": ["i_l (inductor/line current, A)", "v_b (DC bus voltage, V)", "sigma (internal integrator state, V*s)",
                   "v_o (converter terminal voltage, V)"],
        "inputs": ["u (outer-loop actuation, u = dv_o/dt, V/s)"],
        "param_help": {
            "R": "Line resistance (Ohm). Benchmark: 0.20",
            "L": "Line inductance (H). Benchmark: 1.5e-3",
            "C": "Bus capacitance (F). Benchmark: 2.5e-3",
            "v_nom": "Nominal bus voltage setpoint (V). Benchmark: 400",
            "P": "Constant-power load (W). Benchmark: 20000",
            "R_v": "Virtual/target droop-like coefficient in the internal target manifold (Ohm). Benchmark: 0.5",
            "K_i": "Internal integrator gain (1/s). Benchmark: 50",
            "k_m": "Transverse contraction rate (1/s). Benchmark: 500",
        },
        "default_operating_input": [0.0],
        "default_state_guess": [50.0, 400.0, 5.0, 410.0],
        "default_manifold_range": [40.0, 500.0],
        "guarantee_scope": "Local exponential stability of the closed-loop equilibrium is established "
                            "when the Routh-Hurwitz conditions on (a2,a1,a0) hold (see "
                            "StabilizingMRCModel.routh_hurwitz_report) -- NOT a certified regional basin "
                            "of attraction, and NOT verified for the finite-bandwidth sampled actuator "
                            "mode (not implemented in this version).",
    },
}


def list_models() -> str:
    """Return a human-readable listing of available models (used by CLI --list-models and error messages)."""
    lines = ["Available system models:"]
    for key, meta in MODEL_METADATA.items():
        lines.append(f"\n  {key}\n    {meta['label']}")
        lines.append(f"    states: {', '.join(meta['states'])}")
        lines.append(f"    inputs: {', '.join(meta['inputs'])}")
        lines.append("    params:")
        for pname, help_text in meta["param_help"].items():
            lines.append(f"      - {pname}: {help_text}")
    return "\n".join(lines)
