# ResearchSwarm

ResearchSwarm is an MVP automated ML researcher system built with a hierarchical agent-swarm architecture.

## Current status

This repository now includes **Phase 1 scaffolding** from `docs/IMPLEMENTATION_PLAN.md`:

- Monorepo folders for web, orchestrator, runner, shared schemas, and local infra.
- FastAPI orchestrator intake/run endpoints with SQLite-backed persistence.
- Mock specialist orchestration loop with timeline and notebook event writing.
- Minimal web intake form and timeline viewer.
- Dev docker-compose stack and CI checks for lint/typecheck/tests.

## Repository organization

- `apps/web` — intake UI and timeline skeleton.
- `services/orchestrator` — FastAPI backend with orchestrator state machine.
- `services/runner` — experiment runner stub interface.
- `libs/schemas` — shared payload schema definitions.
- `infra/dev` — local docker-compose setup.
- `docs/IMPLEMENTATION_PLAN.md` — end-to-end technical and product plan.
- `docs/AGENT_SYSTEM_DESIGN.md` — hierarchical swarm design and agent contracts.

## Quick start

```bash
make dev
```

Then open:
- web UI: http://localhost:3000
- orchestrator API docs: http://localhost:8000/docs
