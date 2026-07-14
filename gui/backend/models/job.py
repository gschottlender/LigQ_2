from enum import Enum
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    partial_results = "partial_results"
    completed = "completed"
    completed_with_warnings = "completed_with_warnings"
    failed = "failed"


class JobProgress(BaseModel):
    step: str
    label: str
    step_index: int = 1
    step_count: int = 1
    percent: int = 0
    current: Optional[int] = None
    total: Optional[int] = None
    unit: Optional[str] = None
    context: Optional[str] = None
    eta_seconds: Optional[int] = None


class JobFailure(BaseModel):
    step: Optional[str] = None
    label: str
    step_index: Optional[int] = None
    step_count: Optional[int] = None
    message: str


class Job(BaseModel):
    job_id: str
    job_type: str  # "search" | "build_database" | "add_representation"
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    elapsed_seconds: Optional[float] = None
    progress_message: str = ""
    progress_percent: Optional[int] = None
    progress: Optional[JobProgress] = None
    output_dir: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    failure: Optional[JobFailure] = None
    completed_queries: list[str] = Field(default_factory=list)
    all_queries: list[str] = Field(default_factory=list)
    n_queries: Optional[int] = None


class AddRepresentationRequest(BaseModel):
    base_name: str
    representation_type: str  # "rdkit" | "huggingface"
    rep_name: str
    n_bits: int = 1024
    rdkit_fp_kind: Optional[str] = None
    model_id: Optional[str] = None
    batch_size: Optional[int] = None
    n_jobs: Optional[int] = None
