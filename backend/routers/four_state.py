"""
Four-state stabilizing-MRC API router.

All routes mount on the app's /api router (prefix /api/four_state). The
heavy mathematics lives in ims_platform.stabilizing (vendored, verified
core); this module only marshals JSON and caches ONE verified dataset per
run so that CSV export and the PDF report use exactly the plotted data.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

# ims_platform is vendored under backend/; ensure it is importable.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from ims_platform.stabilizing import (  # noqa: E402
    MODEL_ID, MODEL_VERSION,
    validate_params, ParameterError,
    compute_equilibrium, compute_stability, run_simulation, dataset_to_csv,
)
from ims_platform.stabilizing.report import build_pdf  # noqa: E402

router = APIRouter(prefix="/four_state", tags=["four_state"])

# In-memory cache of verified datasets (single source for plots/CSV/PDF).
_RUN_CACHE: Dict[str, Dict] = {}
_MAX_RUNS = 32


class Params(BaseModel):
    R: float
    L: float
    C: float
    P: float
    v_nom: float
    R_v: float
    K_i: float
    k_m: float


class Disturbance(BaseModel):
    P_nom: float
    P_disturbed: float
    t_start: float
    t_clear: float


class Perturbation(BaseModel):
    i_l: float = 0.0
    v_b: float = 0.0
    sigma: float = 0.0
    v_o: float = 0.0


class SimRequest(BaseModel):
    params: Params
    t_end: float = 0.1
    n_eval: int = 2000
    rtol: float = 1e-8
    atol: float = 1e-10
    method: str = "RK45"
    initial_perturbation: Perturbation = Field(default_factory=Perturbation)
    disturbance: Optional[Disturbance] = None


def _cache(dataset: Dict) -> None:
    _RUN_CACHE[dataset["run_id"]] = dataset
    if len(_RUN_CACHE) > _MAX_RUNS:
        # drop oldest inserted
        oldest = next(iter(_RUN_CACHE))
        _RUN_CACHE.pop(oldest, None)


@router.get("/model")
def model_info():
    from ims_platform.stabilizing.workflow import PARAM_SCHEMA
    return {"model_id": MODEL_ID, "model_version": MODEL_VERSION, "param_schema": PARAM_SCHEMA}


@router.post("/validate")
def validate(params: Params):
    try:
        normalized = validate_params(params.model_dump())
        return {"valid": True, "errors": [], "params": normalized}
    except ParameterError as e:
        return {"valid": False, "errors": e.errors, "params": None}


@router.post("/equilibrium")
def equilibrium(params: Params):
    try:
        return compute_equilibrium(params.model_dump())
    except ParameterError as e:
        raise HTTPException(status_code=422, detail={"errors": e.errors})


@router.post("/stability")
def stability(params: Params):
    try:
        return compute_stability(params.model_dump())
    except ParameterError as e:
        raise HTTPException(status_code=422, detail={"errors": e.errors})


@router.post("/simulate")
def simulate(req: SimRequest):
    cfg = {
        "t_end": req.t_end, "n_eval": req.n_eval, "rtol": req.rtol,
        "atol": req.atol, "method": req.method,
        "initial_perturbation": req.initial_perturbation.model_dump(),
        "disturbance": req.disturbance.model_dump() if req.disturbance else None,
    }
    try:
        dataset = run_simulation(req.params.model_dump(), cfg)
    except ParameterError as e:
        raise HTTPException(status_code=422, detail={"errors": e.errors})
    _cache(dataset)
    return dataset


@router.get("/export/csv/{run_id}")
def export_csv(run_id: str):
    ds = _RUN_CACHE.get(run_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="run_id not found (re-run simulation)")
    csv = dataset_to_csv(ds)
    return Response(
        content=csv, media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="four_state_{run_id[:8]}.csv"'},
    )


@router.get("/report/pdf/{run_id}")
def export_pdf(run_id: str):
    ds = _RUN_CACHE.get(run_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="run_id not found (re-run simulation)")
    pdf = build_pdf(ds)
    return Response(
        content=pdf, media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="four_state_report_{run_id[:8]}.pdf"'},
    )
