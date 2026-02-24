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
