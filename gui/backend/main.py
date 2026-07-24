from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from contextlib import asynccontextmanager

from core import state
from core.config import (
    ALLOWED_ORIGINS,
    DATABASES_DIR,
    PIPELINE_ROOT,
    RESULTS_DIR,
    STATE_DIR,
    TEMP_RESULTS_DIR,
    UPLOADS_DIR,
)
from routers import databases, files, jobs, results, setup, system
from services.job_runner import (
    cleanup_stale_resource_jobs,
    cleanup_stale_web_search_jobs,
    start_worker,
    stop_worker,
)
from services.web_access import prepare_web_session, set_web_session_cookie
from services.web_cleanup import start_web_cleanup, stop_web_cleanup

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    for directory in (
        DATABASES_DIR,
        RESULTS_DIR,
        UPLOADS_DIR,
        TEMP_RESULTS_DIR,
        STATE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    await state.initialize()
    await cleanup_stale_resource_jobs()
    await cleanup_stale_web_search_jobs()
    await start_worker()
    await start_web_cleanup()
    try:
        yield
    finally:
        await stop_web_cleanup()
        await stop_worker()


app = FastAPI(
    title="LigQ 2 API",
    version="0.2.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def anonymous_web_session(request: Request, call_next):
    new_session_token = prepare_web_session(request)
    response = await call_next(request)
    set_web_session_cookie(response, new_session_token)
    return response

app.include_router(databases.router)
app.include_router(jobs.router)
app.include_router(results.router)
app.include_router(results.history_router)
app.include_router(files.router)
app.include_router(setup.router)
app.include_router(system.router)


@app.get("/api/health")
async def health():
    return {"status": "ok", "pipeline_root": str(PIPELINE_ROOT)}


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "An unexpected error occurred.",
            "details": None,
        },
    )
