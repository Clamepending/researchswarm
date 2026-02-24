# ResearchSwarm

ResearchSwarm is an MVP automated ML researcher system built with a hierarchical agent-swarm architecture.

## Current status

This repository is currently in **Phase 0 (Planning)**. The first deliverable is a highly detailed implementation plan before any production build-out.

## Planned product capabilities

- UI to submit:
  - dataset (required)
  - optional GitHub codebase
  - optional PDF paper(s)
  - optional research question
- Multi-agent decomposition of research tasks (literature review, hypothesis generation, experiment design, implementation, evaluation, reporting).
- Iterative self-improvement loop with checkpointed findings and informative failure tracking.
- Rigorous testing and reproducibility controls for generated code and claims.
- First toy benchmark focus: image generation experimentation on ImageNet with systematic ablations (noise schedules, integration methods, and related modeling choices).

## Repository organization

- `docs/IMPLEMENTATION_PLAN.md` — end-to-end technical and product plan.
- `docs/AGENT_SYSTEM_DESIGN.md` — detailed hierarchical swarm design and agent contracts.
- `specs/` — future home for API contracts, schemas, and evaluation specs.

## Next action

Implement Phase 1 scaffolding exactly as specified in `docs/IMPLEMENTATION_PLAN.md`.
