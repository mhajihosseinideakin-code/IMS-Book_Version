"""
case.schema
-----------

The user-facing "case file" schema: everything a person needs to describe
about their power system, disturbance, and analysis settings, expressed as
plain data (YAML or JSON) rather than Python code. This is the input
contract for `case.runner.CaseRunner`.

A minimal case file only needs `model` and `name`; every other field has a
sensible default so a first-time user can get a complete analysis from a
handful of lines, and override only what they care about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .registry import MODEL_REGISTRY, MODEL_METADATA, list_models


class CaseValidationError(ValueError):
    """Raised when a case file is malformed or references an unknown model."""


@dataclass
class ManifoldSpec:
    sweep_param_range: Optional[List[float]] = None  # defaults to model metadata if omitted
    sweep_points: int = 60
    keep_unstable: bool = True

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ManifoldSpec":
        d = d or {}
        return cls(
            sweep_param_range=d.get("sweep_param_range"),
            sweep_points=int(d.get("sweep_points", 60)),
            keep_unstable=bool(d.get("keep_unstable", True)),
        )


@dataclass
class DisturbanceSpec:
    #: "offset" = added to the nominal equilibrium; "absolute" = used as-is
    type: str = "offset"
    value: List[float] = field(default_factory=lambda: [0.0, 0.0])
    t_horizon: float = 10.0

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "DisturbanceSpec":
        d = d or {}
        dtype = d.get("type", "offset")
        if dtype not in ("offset", "absolute"):
            raise CaseValidationError(f"disturbance.type must be 'offset' or 'absolute', got '{dtype}'")
        return cls(type=dtype, value=list(d.get("value", [0.0, 0.0])), t_horizon=float(d.get("t_horizon", 10.0)))


@dataclass
class RecoverabilitySpec:
    enabled: bool = True
    radius: float = 1.0
    n_samples: int = 150
    t_horizon: float = 10.0
    recovery_tol: float = 0.1
    radius_sweep: Optional[List[float]] = None
    radius_sweep_samples: int = 30

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "RecoverabilitySpec":
        d = d or {}
        return cls(
            enabled=bool(d.get("enabled", True)),
            radius=float(d.get("radius", 1.0)),
            n_samples=int(d.get("n_samples", 150)),
            t_horizon=float(d.get("t_horizon", 10.0)),
            recovery_tol=float(d.get("recovery_tol", 0.1)),
            radius_sweep=d.get("radius_sweep"),
            radius_sweep_samples=int(d.get("radius_sweep_samples", 30)),
        )


@dataclass
class ControlSpec:
    enabled: bool = False
    Q_diag: Optional[List[float]] = None  # defaults to identity if omitted
    R_diag: Optional[List[float]] = None  # defaults to 0.1*identity if omitted
    u_min: Optional[List[float]] = None
    u_max: Optional[List[float]] = None

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ControlSpec":
        d = d or {}
        return cls(
            enabled=bool(d.get("enabled", False)),
            Q_diag=d.get("Q_diag"),
            R_diag=d.get("R_diag"),
            u_min=d.get("u_min"),
            u_max=d.get("u_max"),
        )


@dataclass
class OutputSpec:
    directory: str = "ims_case_output"
    formats: List[str] = field(default_factory=lambda: ["png", "md", "html"])

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "OutputSpec":
        d = d or {}
        return cls(
            directory=str(d.get("directory", "ims_case_output")),
            formats=list(d.get("formats", ["png", "md", "html"])),
        )


@dataclass
class Case:
    """A complete, validated description of one IMS analysis run."""
    model: str
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    nominal_input: Optional[List[float]] = None
    state_guess: Optional[List[float]] = None
    manifold: ManifoldSpec = field(default_factory=ManifoldSpec)
    disturbance: DisturbanceSpec = field(default_factory=DisturbanceSpec)
    recoverability: RecoverabilitySpec = field(default_factory=RecoverabilitySpec)
    control: ControlSpec = field(default_factory=ControlSpec)
    output: OutputSpec = field(default_factory=OutputSpec)

    @classmethod
    def from_dict(cls, d: dict) -> "Case":
        if "model" not in d:
            raise CaseValidationError(f"case file missing required field 'model'.\n\n{list_models()}")
        model = d["model"]
        if model not in MODEL_REGISTRY:
            raise CaseValidationError(
                f"unknown model '{model}'.\n\n{list_models()}"
            )
        meta = MODEL_METADATA[model]

        params = dict(d.get("params", {}))
        unknown = set(params) - set(meta["param_help"])
        if unknown:
            raise CaseValidationError(
                f"unknown parameter(s) {sorted(unknown)} for model '{model}'. "
                f"Valid parameters: {sorted(meta['param_help'])}"
            )

        return cls(
            model=model,
            name=d.get("name", model),
            params=params,
            nominal_input=d.get("nominal_input"),
            state_guess=d.get("state_guess"),
            manifold=ManifoldSpec.from_dict(d.get("manifold")),
            disturbance=DisturbanceSpec.from_dict(d.get("disturbance")),
            recoverability=RecoverabilitySpec.from_dict(d.get("recoverability")),
            control=ControlSpec.from_dict(d.get("control")),
            output=OutputSpec.from_dict(d.get("output")),
        )

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "name": self.name,
            "params": self.params,
            "nominal_input": self.nominal_input,
            "state_guess": self.state_guess,
            "manifold": vars(self.manifold),
            "disturbance": vars(self.disturbance),
            "recoverability": vars(self.recoverability),
            "control": vars(self.control),
            "output": vars(self.output),
        }
