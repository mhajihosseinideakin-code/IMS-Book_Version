"""
case.wizard
-----------

An interactive, plain-language terminal wizard: the user answers a series
of prompts describing their power system, disturbance, and analysis
preferences, and the wizard builds a `Case` and (optionally) runs the full
analysis immediately, or saves the answers as a reusable case file.

This is the "just let me type in my system's numbers" entry point, for
users who would rather not hand-edit a YAML file.
"""

from __future__ import annotations

import sys
from typing import List, Optional

from .schema import Case, ManifoldSpec, DisturbanceSpec, RecoverabilitySpec, ControlSpec, OutputSpec
from .registry import MODEL_METADATA, list_models


def _ask(prompt: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val if val else (default or "")


def _ask_float(prompt: str, default: float) -> float:
    raw = _ask(prompt, str(default))
    try:
        return float(raw)
    except ValueError:
        print(f"  (couldn't parse '{raw}' as a number, using default {default})")
        return default


def _ask_int(prompt: str, default: int) -> int:
    raw = _ask(prompt, str(default))
    try:
        return int(raw)
    except ValueError:
        print(f"  (couldn't parse '{raw}' as an integer, using default {default})")
        return default


def _ask_bool(prompt: str, default: bool) -> bool:
    raw = _ask(prompt + " (y/n)", "y" if default else "n").lower()
    return raw.startswith("y")


def _ask_vector(prompt: str, dim: int, labels: List[str], default: List[float]) -> List[float]:
    print(f"{prompt} ({dim} value(s): {', '.join(labels)})")
    out = []
    for i in range(dim):
        d = default[i] if i < len(default) else 0.0
        out.append(_ask_float(f"  {labels[i] if i < len(labels) else f'value {i+1}'}", d))
    return out


def run_wizard(stream=sys.stdout) -> Case:
    """Run the interactive wizard and return the resulting Case. Does not execute the analysis."""
    print("=" * 70)
    print("IMS Platform — Interactive System Setup")
    print("=" * 70)
    print("\nAnswer the questions below to describe your power system. Press")
    print("Enter to accept the default shown in [brackets].\n")

    print(list_models())
    print()
    model = _ask("Which system model do you want to analyse?", "grid_forming_inverter")
    while model not in MODEL_METADATA:
        print(f"'{model}' is not a recognised model.")
        model = _ask("Which system model do you want to analyse?", "grid_forming_inverter")
    meta = MODEL_METADATA[model]
    n_states = len(meta["states"])
    n_inputs = len(meta["inputs"])

    name = _ask("Give this study a name", f"My {model} study")

    print(f"\n--- Model parameters for '{model}' (press Enter to keep the physical default) ---")
    params = {}
    for pname, help_text in meta["param_help"].items():
        print(f"  {pname}: {help_text}")
        raw = _ask(f"    {pname} (blank = model default)", "")
        if raw:
            try:
                params[pname] = float(raw)
            except ValueError:
                print(f"    (couldn't parse '{raw}', skipping)")

    print("\n--- Nominal operating point ---")
    nominal_input = _ask_vector("Nominal input / setpoint", n_inputs, meta["inputs"], meta["default_operating_input"])
    state_guess = _ask_vector("Initial guess for the operating equilibrium (any reasonable point works)",
                               n_states, meta["states"], meta["default_state_guess"])

    print("\n--- Intrinsic manifold sweep ---")
    default_range = meta["default_manifold_range"]
    lo = _ask_float("Manifold sweep parameter minimum", default_range[0])
    hi = _ask_float("Manifold sweep parameter maximum", default_range[1])
    sweep_points = _ask_int("Number of manifold sample points", 60)

    print("\n--- Disturbance to analyse ---")
    print("  1) Offset from the nominal operating point (recommended)")
    print("  2) Absolute post-disturbance state")
    dtype_choice = _ask("Disturbance type (1 or 2)", "1")
    dtype = "offset" if dtype_choice.strip() != "2" else "absolute"
    dist_value = _ask_vector(
        "Disturbance " + ("offset" if dtype == "offset" else "absolute state"),
        n_states, meta["states"], [0.0] * n_states
    )
    dist_horizon = _ask_float("Simulation horizon for this disturbance (s)", 10.0)

    print("\n--- Recoverability assessment (Monte-Carlo sweep around the operating point) ---")
    rec_enabled = _ask_bool("Run a recoverability assessment?", True)
    radius = n_samples = t_horizon = tol = None
    radius_sweep = None
    if rec_enabled:
        radius = _ask_float("Disturbance sampling radius", 1.0)
        n_samples = _ask_int("Number of Monte-Carlo samples", 150)
        t_horizon = _ask_float("Simulation horizon per sample (s)", 10.0)
        tol = _ask_float("Manifold-residual tolerance to count as 'recovered'", 0.1)
        if _ask_bool("Also compute a recoverability-vs-radius curve?", True):
            radius_sweep = [round(radius * f, 3) for f in (0.2, 0.4, 0.6, 0.8, 1.0)]

    print("\n--- Manifold-Reshaping Control (MRC) ---")
    control_enabled = _ask_bool("Design and evaluate Manifold-Reshaping Control?", False)

    out_dir = _ask("Output directory for plots and report", f"ims_output_{model}")

    case = Case(
        model=model,
        name=name,
        params=params,
        nominal_input=nominal_input,
        state_guess=state_guess,
        manifold=ManifoldSpec(sweep_param_range=[lo, hi], sweep_points=sweep_points, keep_unstable=True),
        disturbance=DisturbanceSpec(type=dtype, value=dist_value, t_horizon=dist_horizon),
        recoverability=RecoverabilitySpec(
            enabled=rec_enabled,
            radius=radius or 1.0, n_samples=n_samples or 150,
            t_horizon=t_horizon or 10.0, recovery_tol=tol or 0.1,
            radius_sweep=radius_sweep,
        ),
        control=ControlSpec(enabled=control_enabled),
        output=OutputSpec(directory=out_dir),
    )

    print("\nSetup complete.\n")
    return case


if __name__ == "__main__":
    c = run_wizard()
    print(c.to_dict())
