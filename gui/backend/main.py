from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import ALLOWED_ORIGINS, PIPELINE_ROOT, UPLOADS_DIR
from routers import databases, files, jobs, results, setup

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(title="LigQ 2 API", version="0.1.0", docs_url="/api/docs", redoc_url="/api/redoc")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(databases.router)
app.include_router(jobs.router)
app.include_router(results.router)
app.include_router(results.history_router)
app.include_router(files.router)
app.include_router(setup.router)


@app.get("/api/health")
async def health():
    return {"status": "ok", "pipeline_root": str(PIPELINE_ROOT)}


@app.on_event("startup")
async def startup():
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


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
