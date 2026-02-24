# Orchestrator Service

FastAPI service for intake, run creation, and specialist orchestration.

## Local run

```bash
pip install -r services/orchestrator/requirements.txt
uvicorn app.main:app --reload --app-dir services/orchestrator
```

## End-to-end MNIST image-generation settings example

1. Create a project: `POST /api/projects`
2. Create a run: `POST /api/projects/{project_id}/runs`
3. Execute candidate search: `POST /api/runs/{run_id}/mnist-imagegen/execute`
4. Inspect timeline: `GET /api/runs/{run_id}/timeline`

The execute endpoint deduplicates identical candidates, validates parameter ranges,
computes deterministic proxy metrics (FID proxy, stability, throughput, memory), and
returns ranked settings plus caveats for reproducibility follow-ups.
