"""
MRC Designer API router (prefix /api/mrc_designer).

Network Builder -> Model Assembly -> IMS Analysis -> [Design MRC] ->
Closed-Loop Verification -> Before/After. The heavy work lives in
ims_platform.mrc_designer; this marshals JSON and reuses the four_state run
cache so CSV/PDF export resolves designer run_ids too.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from ims_platform.mrc_designer import (  # noqa: E402
    feasibility_report, synthesize_mrc, verify_closed_loop, before_after, MRCNotEstablished,
)
from ims_platform.mrc_designer.designer import get_designer_model_ids, _DESIGNER_MODELS  # noqa: E402
from ims_platform.models.stabilizing_mrc import StabilizingMRCModel  # noqa: E402
from routers.four_state import _cache as _four_state_cache  # noqa: E402

router = APIRouter(prefix="/mrc_designer", tags=["mrc_designer"])


class DesignRequest(BaseModel):
    model_id: str = "stabilizing_mrc"
    params: Dict[str, float]
    k_m: Optional[float] = None


class SimDesignRequest(DesignRequest):
    t_end: float = 0.06
    n_eval: int = 2000
    rtol: float = 1e-8
    atol: float = 1e-10
    method: str = "RK45"
    initial_perturbation: Dict[str, float] = {}
    disturbance: Optional[Dict[str, float]] = None


def _sim_cfg(req: "SimDesignRequest") -> Dict[str, Any]:
    return {"t_end": req.t_end, "n_eval": req.n_eval, "rtol": req.rtol, "atol": req.atol,
            "method": req.method, "initial_perturbation": req.initial_perturbation,
            "disturbance": req.disturbance}


@router.get("/models")
def models():
    out = []
    for mid in get_designer_model_ids():
        spec = _DESIGNER_MODELS[mid]
        out.append({"model_id": mid, "title": spec["title"], "manifold_roles": spec["manifold_roles"]})
    return {"supported_models": out,
            "workflow": ["Network Builder", "Model Assembly", "IMS Analysis", "Design MRC",
                         "Closed-Loop Verification", "Before/After Comparison"]}


@router.get("/default_params/{model_id}")
def default_params(model_id: str):
    if model_id == "stabilizing_mrc":
        return StabilizingMRCModel().default_params()
    raise HTTPException(status_code=404, detail=f"no default params for '{model_id}'")


@router.post("/feasibility")
def feasibility(req: DesignRequest):
    if req.model_id not in _DESIGNER_MODELS:
        raise HTTPException(status_code=404, detail=f"unsupported model '{req.model_id}'")
    system = _DESIGNER_MODELS[req.model_id]["factory"](req.params)
    x_star = system.equilibrium_closed_form(req.params) if hasattr(system, "equilibrium_closed_form") \
        else system.find_equilibrium(system.initial_guess(), with_eigs=False).x_star
    return feasibility_report(system, x_star, req.params)


@router.post("/synthesize")
def synthesize(req: DesignRequest):
    try:
        design = synthesize_mrc(req.model_id, req.params, k_m=req.k_m)
    except MRCNotEstablished as e:
        # Explicit, non-error signal: MRC not established (not a 500).
        return {"mrc_established": False, "reason": str(e)}
    return {"mrc_established": True, **{k: v for k, v in design.items() if not k.startswith("_")}}


@router.post("/verify")
def verify(req: SimDesignRequest):
    try:
        env = verify_closed_loop(req.model_id, req.params, _sim_cfg(req), k_m=req.k_m)
    except MRCNotEstablished as e:
        raise HTTPException(status_code=422, detail={"mrc_established": False, "reason": str(e)})
    _maybe_cache(env)
    return env


@router.post("/before_after")
def compare(req: SimDesignRequest):
    try:
        return before_after(req.model_id, req.params, _sim_cfg(req), k_m=req.k_m)
    except MRCNotEstablished as e:
        raise HTTPException(status_code=422, detail={"mrc_established": False, "reason": str(e)})


def _maybe_cache(env: Dict[str, Any]) -> None:
    if not env.get("run_id"):
        return
    traj = env.get("trajectories") or {}
    res = env.get("residual_trajectory") or {}
    contr = env.get("contraction_evidence") or {}
    _four_state_cache({
        "run_id": env["run_id"], "t": traj.get("t", []),
        "i_l": traj.get("i_l", []), "v_b": traj.get("v_b", []), "sigma": traj.get("sigma", []),
        "v_o": traj.get("v_o", []), "e_sigma": res.get("e_sigma", []),
        "e_sigma_theory": res.get("e_sigma_theory", []), "cpl_power": traj.get("cpl_power", []),
        "disturbance_active": [], "x_star": traj.get("x_star", {}),
        "e_sigma_0": (res.get("e_sigma") or [0.0])[0] if res.get("e_sigma") else 0.0,
        "residual_contraction": {"max_abs_error": float("nan"),
                                 "max_rel_error": contr.get("max_rel_error", float("nan")),
                                 "k_m": (env.get("design") or {}).get("k_m", float("nan")),
                                 "note": contr.get("identity", "")},
        "events": traj.get("events", []), "status": env.get("recovery_status", {}),
        "provenance": env.get("provenance", {}),
    })
