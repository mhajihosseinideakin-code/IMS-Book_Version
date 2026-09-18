"""
case.loader
-----------

Reads a user-authored case file (YAML preferred, JSON also accepted) into a
validated `Case` object, and can emit a commented starter template for a
given model so a new user has something concrete to edit rather than a
blank page.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from .schema import Case, CaseValidationError
from .registry import MODEL_METADATA, list_models

try:
    import yaml
    _HAS_YAML = True
except ImportError:  # pragma: no cover
    _HAS_YAML = False


def load_case(path: str) -> Case:
    """Load a case file (.yaml/.yml/.json) from disk into a validated Case."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"case file not found: {path}")

    with open(path, "r") as f:
        text = f.read()

    ext = os.path.splitext(path)[1].lower()
    if ext in (".yaml", ".yml"):
        if not _HAS_YAML:
            raise RuntimeError("PyYAML is not installed; use a .json case file instead, or `pip install pyyaml`.")
        data = yaml.safe_load(text)
    elif ext == ".json":
        data = json.loads(text)
    else:
        # best-effort: try YAML first (a superset of JSON), then JSON
        if _HAS_YAML:
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)

    if not isinstance(data, dict):
        raise CaseValidationError(f"case file '{path}' did not parse to a mapping/object at the top level.")

    try:
        return Case.from_dict(data)
    except CaseValidationError:
        raise
    except Exception as e:  # re-wrap for a consistent, actionable error surface
        raise CaseValidationError(f"error parsing case file '{path}': {e}") from e


def save_case_template(model: str, path: str) -> None:
    """
    Write a starter case file for `model` to `path` (.yaml or .json,
    inferred from the extension), pre-filled with the model's default
    parameters and sensible analysis settings, ready for the user to edit.
    """
    if model not in MODEL_METADATA:
        raise CaseValidationError(f"unknown model '{model}'.\n\n{list_models()}")
    meta = MODEL_METADATA[model]

    template = {
        "model": model,
        "name": f"My {meta['label'].split('(')[0].strip()} Study",
        "params": {},  # leave empty to use model defaults; override only what you need
        "nominal_input": meta["default_operating_input"],
        "state_guess": meta["default_state_guess"],
        "manifold": {
            "sweep_param_range": meta["default_manifold_range"],
            "sweep_points": 60,
            "keep_unstable": True,
        },
        "disturbance": {
            "type": "offset",
            "value": [0.0] * len(meta["states"]),
            "t_horizon": 10.0,
        },
        "recoverability": {
            "enabled": True,
            "radius": 1.0,
            "n_samples": 150,
            "t_horizon": 10.0,
            "recovery_tol": 0.1,
        },
        "control": {
            "enabled": False,
            "Q_diag": None,
            "R_diag": None,
            "u_min": None,
            "u_max": None,
        },
        "output": {
            "directory": f"ims_output_{model}",
            "formats": ["png", "md", "html"],
        },
    }

    ext = os.path.splitext(path)[1].lower()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        if ext in (".yaml", ".yml") and _HAS_YAML:
            f.write(f"# IMS Platform case file — {meta['label']}\n")
            f.write(f"# States : {', '.join(meta['states'])}\n")
            f.write(f"# Inputs : {', '.join(meta['inputs'])}\n")
            f.write("# Editable parameters (add under `params:` to override model defaults):\n")
            for pname, help_text in meta["param_help"].items():
                f.write(f"#   {pname}: {help_text}\n")
            f.write("\n")
            yaml.safe_dump(template, f, sort_keys=False)
        else:
            json.dump(template, f, indent=2)
