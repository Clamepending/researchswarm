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
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS experiment_runs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id TEXT NOT NULL,
          experiment_family TEXT NOT NULL,
          config_json TEXT NOT NULL,
          metrics_json TEXT NOT NULL,
          status TEXT NOT NULL,
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


def update_run_state(run_id: UUID, *, status: str | None = None, stage: str | None = None) -> None:
    if status is None and stage is None:
        return

    conn = _conn()
    if status is not None and stage is not None:
        conn.execute("UPDATE research_runs SET status = ?, stage = ? WHERE id = ?", (status, stage, str(run_id)))
    elif status is not None:
        conn.execute("UPDATE research_runs SET status = ? WHERE id = ?", (status, str(run_id)))
    else:
        conn.execute("UPDATE research_runs SET stage = ? WHERE id = ?", (stage, str(run_id)))
    conn.commit()
    conn.close()


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


def add_experiment_run(run_id: UUID, experiment_family: str, config_json: str, metrics_json: str, status: str = "completed") -> None:
    conn = _conn()
    conn.execute(
        "INSERT INTO experiment_runs (run_id, experiment_family, config_json, metrics_json, status, created_at) VALUES (?, ?, ?, ?, ?, datetime('now'))",
        (str(run_id), experiment_family, config_json, metrics_json, status),
    )
    conn.commit()
    conn.close()
