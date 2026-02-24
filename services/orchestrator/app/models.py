from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl


class ProjectCreate(BaseModel):
    name: str = Field(min_length=2)
    dataset_handle: str = Field(min_length=1)
    github_repo_url: HttpUrl | None = None
    seed_question: str | None = None


class Project(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    dataset_handle: str
    github_repo_url: HttpUrl | None = None
    seed_question: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ResearchRun(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    stage: Literal["intake", "discovery", "question_generation", "planning", "execution", "evaluation", "reporting"] = "intake"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TimelineEvent(BaseModel):
    run_id: UUID
    event_type: Literal["stage_transition", "agent_note", "failure", "retry"]
    message: str
    confidence: float = Field(ge=0, le=1, default=0.5)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Citation(BaseModel):
    source_type: Literal["paper", "repo", "experiment", "note"]
    source_ref: str
    snippet: str


class NotebookEntry(BaseModel):
    run_id: UUID
    author_agent: str
    summary: str
    decision: str
    citations: list[Citation] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)




class TinyImagenetCandidate(BaseModel):
    noise_schedule: Literal["linear", "cosine", "sigmoid"]
    prediction_target: Literal["epsilon", "v_prediction"]
    learning_rate: float = Field(ge=1e-4, le=0.03)
    hidden_size: int = Field(ge=4, le=64)
    train_steps: int = Field(ge=10, le=160)


class TinyImagenetPlanRequest(BaseModel):
    objective: str = Field(min_length=10)
    candidates: list[TinyImagenetCandidate] = Field(min_length=1)


class TinyImagenetEvaluation(BaseModel):
    rank: int
    noise_schedule: str
    prediction_target: str
    learning_rate: float
    hidden_size: int
    train_steps: int
    train_loss: float
    val_loss: float
    stability_score: float
    throughput: float
    score: float


class TinyImagenetBatchReport(BaseModel):
    run_id: UUID
    objective: str
    total_candidates: int
    evaluated_candidates: int
    rejected_candidates: int
    best: TinyImagenetEvaluation
    rankings: list[TinyImagenetEvaluation]
    caveats: list[str]
class MnistImageGenCandidate(BaseModel):
    noise_schedule: Literal["linear", "cosine", "sigmoid"]
    sampler: Literal["ddpm", "ddim", "heun"]
    guidance_scale: float = Field(ge=0.1, le=12.0)
    learning_rate: float = Field(ge=1e-5, le=5e-3)
    ema_decay: float = Field(ge=0.9, le=0.9999)
    grad_clip: float = Field(ge=0.1, le=5.0)


class MnistImageGenPlanRequest(BaseModel):
    objective: str = Field(min_length=10)
    candidates: list[MnistImageGenCandidate] = Field(min_length=1)


class MnistImageGenEvaluation(BaseModel):
    rank: int
    noise_schedule: str
    sampler: str
    guidance_scale: float
    learning_rate: float
    ema_decay: float
    grad_clip: float
    fid_proxy: float
    stability_score: float
    throughput: float
    memory_gb: float
    score: float


class MnistImageGenBatchReport(BaseModel):
    run_id: UUID
    objective: str
    total_candidates: int
    evaluated_candidates: int
    rejected_candidates: int
    best: MnistImageGenEvaluation
    rankings: list[MnistImageGenEvaluation]
    caveats: list[str]
