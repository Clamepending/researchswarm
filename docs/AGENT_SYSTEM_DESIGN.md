# Agent System Design (Hierarchy + Responsibilities)

## Top-level hierarchy

- **Executive Orchestrator**
  - global planner
  - budget manager
  - stage transition controller

- **Research Director**
  - hypothesis portfolio manager
  - prioritization and pruning

- **Specialists**
  - Literature Review Agent
  - Codebase Recon Agent
  - Methodology Critic Agent
  - Experiment Designer Agent
  - Implementation Agent
  - Evaluation & Statistics Agent
  - Reproducibility Auditor Agent
  - Notebook Curator Agent

## Control flow

1. Intake validated.
2. Director requests discovery reports.
3. Critic filters candidate questions.
4. Designer emits experiment graph.
5. Implementation agent materializes runnable configs/code deltas.
6. Runner executes.
7. Evaluator scores outcomes.
8. Auditor verifies reproducibility and reporting integrity.
9. Orchestrator decides continue/stop.

## Shared interfaces

Each agent must implement:
- `plan(task) -> subtask[]`
- `execute(task) -> result`
- `self_critique(result) -> critique`
- `handoff(result, target_agent)`

## Decision policies

- **Continue policy:** expected information gain > threshold and budget remains.
- **Stop policy:** worth-showing criteria satisfied OR budget depleted OR convergence detected.

## Logging requirements

Each agent logs:
- decision summary,
- evidence references,
- confidence,
- known failure modes,
- next action recommendation.
