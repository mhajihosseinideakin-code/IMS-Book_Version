"""
Shared IMS analysis-framework API router (prefix /api/analysis).

Generic endpoints that drive ANY registered analysis through the standard
workflow. The Four-State MRC delegates to the same verified workflow used by
/api/four_state/*, and reuses that router's run cache so CSV/PDF export
(GET /api/four_state/export/csv|report/pdf/{run_id}) works for framework runs
too -- one dataset, one source of truth.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from ims_platform.framework import list_descriptors, get_analysis, STANDARD_WORKFLOW_STAGES  # noqa: E402
from ims_platform.framework.registry import get_descriptor  # noqa: E402
from routers.four_state import _cache as _four_state_cache  # noqa: E402  (reuse the single run cache)

router = APIRouter(prefix="/analysis", tags=["analysis"])


class RunConfig(BaseModel):
    config: Dict[str, Any] = {}


@router.get("/registry")
def registry():
    return {"workflow_stages": STANDARD_WORKFLOW_STAGES, "analyses": list_descriptors()}


@router.get("/{analysis_id}/descriptor")
def descriptor(analysis_id: str):
    try:
        return get_descriptor(analysis_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown analysis '{analysis_id}'")


@router.get("/{analysis_id}/default_config")
def default_config(analysis_id: str):
    try:
        return get_analysis(analysis_id).default_config()
    except KeyError:
        raise HTTPException(status_code=404, detail=f"analysis '{analysis_id}' is not runnable via the framework")


@router.post("/{analysis_id}/validate")
def validate(analysis_id: str, body: RunConfig):
    try:
        analysis = get_analysis(analysis_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"analysis '{analysis_id}' is not runnable via the framework")
    errors = analysis.validate(body.config)
    return {"valid": not errors, "errors": errors}


@router.post("/{analysis_id}/run")
def run(analysis_id: str, body: RunConfig):
    try:
        analysis = get_analysis(analysis_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"analysis '{analysis_id}' is not runnable via the framework")
    errors = analysis.validate(body.config)
    if errors:
        raise HTTPException(status_code=422, detail={"errors": errors})
    envelope = analysis.run(body.config)
    # Reuse the four_state run cache so CSV/PDF export endpoints resolve this run_id.
    if envelope.get("run_id") and envelope.get("provenance") is not None:
        prov = envelope["provenance"]
        traj = envelope.get("trajectories") or {}
        res = envelope.get("residual_trajectory") or {}
        contr = envelope.get("contraction_evidence") or {}
        _four_state_cache({
            "run_id": envelope["run_id"],
            "t": traj.get("t", []),
            "i_l": traj.get("i_l", []), "v_b": traj.get("v_b", []),
            "sigma": traj.get("sigma", []), "v_o": traj.get("v_o", []),
            "e_sigma": res.get("e_sigma", []), "e_sigma_theory": res.get("e_sigma_theory", []),
            "cpl_power": traj.get("cpl_power", []),
            "disturbance_active": traj.get("disturbance_active", []),
            "x_star": traj.get("x_star", {}),
            "e_sigma_0": (envelope.get("manifold") or {}).get("e_sigma_0", 0.0),
            "residual_contraction": {
                "max_abs_error": contr.get("max_abs_error", float("nan")),
                "max_rel_error": contr.get("max_rel_error", float("nan")),
                "k_m": contr.get("rate_k_m", float("nan")),
                "note": contr.get("identity", ""),
            },
            "events": traj.get("events", []),
            "status": envelope.get("recovery_status", {}),
            "provenance": prov,
        })
    return envelope
