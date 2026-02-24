from __future__ import annotations

from uuid import UUID

from fastapi import FastAPI, HTTPException

from .models import MnistImageGenPlanRequest, ProjectCreate
from .orchestrator import OrchestratorStateMachine
from .storage import create_project, create_run, init_db, list_timeline

app = FastAPI(title="ResearchSwarm Orchestrator", version="0.2.0")
state_machine = OrchestratorStateMachine()


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/projects")
def create_project_endpoint(payload: ProjectCreate) -> dict:
    project = create_project(payload)
    return {"project": project.model_dump(mode="json")}


@app.post("/api/projects/{project_id}/runs")
def create_run_endpoint(project_id: UUID) -> dict:
    run = create_run(project_id)
    objective = "Generate and evaluate high-value hypotheses from provided context"
    state_machine.run_discovery(run.id, objective)
    return {"run": run.model_dump(mode="json")}


@app.post("/api/runs/{run_id}/mnist-imagegen/execute")
def execute_mnist_imagegen(run_id: UUID, payload: MnistImageGenPlanRequest) -> dict:
    try:
        report = state_machine.run_mnist_imagegen_batch(
            run_id=run_id,
            objective=payload.objective,
            candidates=payload.candidates,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"report": report.model_dump(mode="json")}


@app.get("/api/runs/{run_id}/timeline")
def timeline(run_id: UUID) -> dict:
    events = [e.model_dump(mode="json") for e in list_timeline(run_id)]
    return {"events": events}
