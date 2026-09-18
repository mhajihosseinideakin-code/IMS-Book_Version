import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List
import uuid
from datetime import datetime


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
from lib.db import client, db, ensure_indexes


# Startup runs before the yield, shutdown after it. Add your own setup/teardown here.
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.index_task = asyncio.create_task(ensure_indexes())  # background: a big index build must not block boot
    yield
    client.close()


# Create the main app without a prefix
app = FastAPI(lifespan=lifespan)

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# Define Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "Hello World"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.model_dump())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

# Four-state stabilizing-MRC routes (mounted on api_router -> /api/four_state/*)
from routers.four_state import router as four_state_router
api_router.include_router(four_state_router)

# Shared IMS analysis framework routes (mounted on api_router -> /api/analysis/*)
from routers.analysis import router as analysis_router
api_router.include_router(analysis_router)

# Include the router in the main app
app.include_router(api_router)

# ----------------------------------------------------------------------
# Preserve the existing Flask IMS Platform (explorer.html + ~60 legacy
# routes) by mounting it as a WSGI sub-application. FastAPI routes above
# take precedence; anything unmatched (/, /assets/*, /api/health,
# /api/projects, /api/iberian/*, ...) is served by the legacy Flask app.
# ----------------------------------------------------------------------
try:
    import sys as _sys
    from pathlib import Path as _Path
    _bd = str(_Path(__file__).resolve().parent)
    if _bd not in _sys.path:
        _sys.path.insert(0, _bd)
    from starlette.middleware.wsgi import WSGIMiddleware
    from ims_platform.server.app import create_app as _create_flask_app
    _flask_app = _create_flask_app()
    # Mount the existing IMS Platform (Flask) as the PRIMARY application at root.
    # This is a catch-all mounted LAST, so the FastAPI routes registered above
    # (/api/four_state/*, /api/status, /api/, /docs) take precedence; every other
    # path (/, /assets/*, /api/health, /api/projects, /api/case_library,
    # /api/iberian/*, /api/gfm_current_limit/*, /api/multi_converter_fault/*,
    # /api/report/*, /api/auth/*) is served by the existing platform unchanged.
    app.mount("/", WSGIMiddleware(_flask_app))
    logging.getLogger(__name__).info("IMS Platform (Flask) mounted at / (primary app)")
except Exception as _e:  # pragma: no cover - legacy app is optional at runtime
    logging.getLogger(__name__).warning("Legacy Flask app not mounted: %s", _e)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
