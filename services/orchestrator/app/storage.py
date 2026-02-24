from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import UUID

from .models import NotebookEntry, Project, ProjectCreate, ResearchRun, TimelineEvent

DB_PATH = Path(__file__).resolve().parent.parent / "researchswarm.db"


def _conn() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    conn = _conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS projects (
          id TEXT PRIMARY KEY,
          name TEXT NOT NULL,
          dataset_handle TEXT NOT NULL,
          github_repo_url TEXT,
          seed_question TEXT,
          created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS research_runs (
          id TEXT PRIMARY KEY,
          project_id TEXT NOT NULL,
          status TEXT NOT NULL,
          stage TEXT NOT NULL,
          created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS timeline_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id TEXT NOT NULL,
          event_type TEXT NOT NULL,
          message TEXT NOT NULL,
          confidence REAL NOT NULL,
          created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS notebook_entries (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id TEXT NOT NULL,
          author_agent TEXT NOT NULL,
          summary TEXT NOT NULL,
          decision TEXT NOT NULL,
          citations_json TEXT NOT NULL,
          created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def create_project(payload: ProjectCreate) -> Project:
    project = Project(**payload.model_dump())
    conn = _conn()
    conn.execute(
        "INSERT INTO projects VALUES (?, ?, ?, ?, ?, ?)",
        (
            str(project.id),
            project.name,
            project.dataset_handle,
            str(project.github_repo_url) if project.github_repo_url else None,
            project.seed_question,
            project.created_at.isoformat(),
        ),
    )
    conn.commit()
    conn.close()
    return project


def create_run(project_id: UUID) -> ResearchRun:
    run = ResearchRun(project_id=project_id)
    conn = _conn()
    conn.execute(
        "INSERT INTO research_runs VALUES (?, ?, ?, ?, ?)",
        (str(run.id), str(run.project_id), run.status, run.stage, run.created_at.isoformat()),
    )
    conn.commit()
    conn.close()
    return run


def add_timeline_event(event: TimelineEvent) -> None:
    conn = _conn()
    conn.execute(
        "INSERT INTO timeline_events (run_id, event_type, message, confidence, created_at) VALUES (?, ?, ?, ?, ?)",
        (str(event.run_id), event.event_type, event.message, event.confidence, event.created_at.isoformat()),
    )
    conn.commit()
    conn.close()


def list_timeline(run_id: UUID) -> list[TimelineEvent]:
    conn = _conn()
    rows = conn.execute(
        "SELECT run_id, event_type, message, confidence, created_at FROM timeline_events WHERE run_id = ? ORDER BY id ASC",
        (str(run_id),),
    ).fetchall()
    conn.close()
    return [TimelineEvent(**dict(row)) for row in rows]


def add_notebook_entry(entry: NotebookEntry) -> None:
    conn = _conn()
    conn.execute(
        "INSERT INTO notebook_entries (run_id, author_agent, summary, decision, citations_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (
            str(entry.run_id),
            entry.author_agent,
            entry.summary,
            entry.decision,
            entry.model_dump_json(include={"citations"}),
            entry.created_at.isoformat(),
        ),
    )
    conn.commit()
    conn.close()
