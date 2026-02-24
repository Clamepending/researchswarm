# ResearchSwarm MVP — Detailed Build Plan

## 1) Product goal and scope

Build an automated ML researcher MVP that can:
1. ingest a dataset and optional context artifacts (GitHub repository, paper PDFs, user question),
2. perform structured literature and codebase review,
3. propose high-quality research questions and hypotheses,
4. plan and execute experiments,
5. evaluate outcomes with statistical and qualitative rigor,
6. iterate until it has reportable results (including informative failures), and
7. produce transparent notes, logs, and reproducible outputs.

### In-scope (MVP)
- Single-user web app.
- Hierarchical agent swarm with orchestrator.
- Async job execution backend.
- Experiment runtime targeting Python ML workloads (PyTorch-first).
- Artifact store for metrics, plots, checkpoints, reports.
- Built-in “research memory” (notes, rationale, decisions, failed attempts).
- Initial benchmark workflow for image-generation experiments on ImageNet subsets.

### Out-of-scope (MVP)
- Multi-tenant enterprise auth.
- Full autonomous cloud infrastructure management.
- Massive-scale cluster scheduling beyond one node or small worker pool.

---

## 2) User experience requirements

## UI inputs
Required:
- Dataset input (upload path, mounted path, or registry handle).

Optional:
- GitHub repo URL (public/private with token config).
- PDF paper uploads.
- User-provided seed question.

## UI outputs
- Real-time run timeline (agent steps, status, failures, retries).
- Hypotheses queue with confidence and rationale.
- Experiment table with config, state, metrics, and links to artifacts.
- Research notebook stream (auto-generated notes + citations + decisions).
- “Results worth showing” dashboard:
  - best runs,
  - negative/failed findings,
  - uncertainty and caveats,
  - reproducibility checklist.

---

## 3) System architecture overview

Use a **hierarchical swarm**:

1. **Executive Orchestrator Agent**
   - Owns global objective and budget.
   - Decomposes work into stages.
   - Decides stop/continue criteria.

2. **Research Director Agent**
   - Converts objective into prioritized research questions.
   - Selects next hypothesis batch.

3. **Specialist Layer (parallel sub-agents)**
   - Literature Review Agent.
   - Codebase Recon Agent.
   - Methodology Critic Agent (taste/quality filter).
   - Experiment Designer Agent.
   - Implementation Agent.
   - Evaluation & Statistics Agent.
   - Reproducibility Auditor Agent.
   - Lab Notebook Curator Agent.

4. **Execution Layer**
   - Tooling adapters for:
     - Git operations,
     - PDF parsing,
     - experiment runner,
     - metric store,
     - report renderer.

5. **Memory Layer**
   - Structured memory (DB tables/doc store).
   - Vector search index for papers/code/docs.
   - Immutable run/event log.

---

## 4) Agent contracts and decomposition protocol

Every agent receives and emits typed payloads.

## Core payload schema (conceptual)
- `task_id`
- `parent_task_id`
- `objective`
- `inputs`
- `constraints`
- `assumptions`
- `acceptance_criteria`
- `artifacts`
- `result_summary`
- `confidence`
- `risks`
- `next_actions`

## Mandatory behaviors
- Explicit uncertainty estimates.
- Explicit evidence references (paper snippets, code lines, experiment IDs).
- “No silent pass”: every completion requires acceptance-criteria check.
- Fallback behavior when tools fail.

## Decomposition pattern
1. Clarify objective.
2. Produce 2–5 subplans.
3. Score subplans by expected information gain / cost.
4. Execute top candidate(s) in parallel where safe.
5. Consolidate and update global belief state.

---

## 5) Research loop lifecycle

1. **Intake phase**
   - Validate dataset availability and schema.
   - Fetch/parse optional GitHub repo.
   - Parse PDFs to chunked citation graph.

2. **Discovery phase**
   - Literature Review Agent extracts methods, unresolved questions, benchmark norms.
   - Codebase Recon Agent inventories reusable components and risks.

3. **Question generation phase**
   - Generate candidate questions.
   - Methodology Critic filters for novelty, tractability, and relevance.
   - Director selects top-K hypotheses.

4. **Experiment planning phase**
   - Designer defines experiment matrix (controls + ablations).
   - Auditor verifies confound controls and compute budget compliance.

5. **Implementation phase**
   - Implementation Agent creates experiment code/config deltas.
   - Automated test suite + smoke training checks must pass.

6. **Execution phase**
   - Run experiments with checkpointing and failure recovery.
   - Stream metrics and artifacts.

7. **Evaluation phase**
   - Statistical comparison vs baselines.
   - Failure taxonomy and root-cause analysis.
   - Update belief/ranking of hypotheses.

8. **Iteration gate**
   - Continue if expected information gain remains high and budget remains.
   - Stop and report when “worth showing” criteria are met.

9. **Final reporting phase**
   - Export report, reproducibility manifest, and machine-readable experiment index.

---

## 6) “Good taste” and “worth showing” criteria

## Good taste heuristics
- Align with active uncertainty in literature.
- Avoid trivial parameter sweeps lacking explanatory value.
- Prefer experiments that isolate one mechanism at a time.
- Favor interpretable comparisons to known baselines.

## Worth-showing threshold (MVP rubric)
At least one of:
- Clear improvement over baseline with robustness checks.
- Strong negative result that falsifies plausible hypothesis.
- Insightful trade-off characterization (quality vs compute, stability vs speed).

And all of:
- Reproducible run manifests.
- Confidence intervals or repeated-run stability evidence.
- Documented caveats.

---

## 7) Toy benchmark plan: ImageNet image generation

## Primary objective
Create an automated loop that explores and compares image-generation training choices on ImageNet (start with subset for MVP).

## Initial experiment families
- Noise schedule variants.
- Integration/sampling method variants.
- Objective weighting variants.
- Training stability interventions (gradient clipping, EMA behavior, LR schedule).

## Minimum baseline set
- Baseline diffusion-like configuration.
- At least 3 controlled ablation families.
- Common metrics: FID proxy, training stability stats, throughput, memory.

## Phased dataset strategy
- Phase A: Tiny ImageNet / ImageNet subset for fast iteration.
- Phase B: larger subset with stricter eval.
- Phase C: optional full-scale runs.

---

## 8) Technical stack proposal

## Frontend
- Next.js + TypeScript.
- Component library: shadcn/ui or equivalent.
- State/query: React Query.
- Real-time updates via SSE or WebSockets.

## Backend
- Python FastAPI service.
- Task queue: Celery or Dramatiq.
- Broker: Redis.
- DB: Postgres.
- Artifact storage: local S3-compatible (MinIO) in dev.

## ML execution
- PyTorch + Hydra for configs.
- Experiment tracking: MLflow or Weights & Biases (toggleable).
- Containerized runners.

## Retrieval and memory
- Document chunking via PyMuPDF + custom parser.
- Embeddings index for paper/code retrieval.
- Structured notes in Postgres + markdown exports.

---

## 9) Data model (MVP entities)

- `projects`
- `research_runs`
- `agent_tasks`
- `hypotheses`
- `experiments`
- `experiment_runs`
- `artifacts`
- `findings`
- `notebook_entries`
- `citations`
- `repro_checks`

Each entity should include timestamps, provenance (`created_by_agent`), and versioned status.

---

## 10) Safety, reliability, and rigor requirements

- Every code change generated by agents must pass:
  - lint,
  - unit tests,
  - smoke train/eval test.
- Experiment claims must cite run IDs and metric sources.
- Automatic detection of p-hacking patterns (excessive selective reruns without disclosure).
- Hard budget controls (GPU hours, wall-clock, max retries).
- Full audit trail for each decision.

---

## 11) Implementation roadmap (execution order)

## Phase 0 — Planning (this deliverable)
- Finalize architecture and contracts.
- Define success metrics and acceptance criteria.

## Phase 1 — Skeleton platform
- Monorepo scaffold:
  - `apps/web`
  - `services/orchestrator`
  - `services/runner`
  - `libs/schemas`
  - `infra/dev`
- Basic UI with intake form and run status page.
- Backend API stubs and DB migrations.

## Phase 2 — Core agent loop
- Implement orchestrator + 3 initial specialists:
  - literature,
  - experiment designer,
  - evaluator.
- Add memory store and notebook writer.

## Phase 3 — Experiment execution integration
- Add training runner interface and baseline experiment templates.
- Add metric ingestion and artifact linking.

## Phase 4 — Iterative autonomy
- Add stop/continue policy.
- Add critique/reflection agent and quality gates.

## Phase 5 — Toy benchmark completion
- Run end-to-end ImageNet subset benchmark.
- Produce first automated research report.

---

## 12) Acceptance criteria for MVP

Functional:
- User can submit dataset + optional repo/pdf/question in UI.
- System generates at least 3 candidate research questions.
- System executes at least 1 full experiment batch autonomously.
- System outputs report with at least 1 positive or informative negative finding.

Quality:
- Re-running the same config reproduces key metrics within tolerance.
- All findings map to citations + run artifacts.
- Notebook timeline reconstructs decisions end-to-end.

Operational:
- One-command local dev startup for web + API + worker + DB.
- Failed tasks recover via retry policy and preserve logs.

---

## 13) Risks and mitigations

- **Risk:** Agent drift into low-value sweeps.
  - **Mitigation:** information-gain scoring + critic veto.
- **Risk:** Fragile generated code.
  - **Mitigation:** strict test gates and staged rollout.
- **Risk:** Cost explosion.
  - **Mitigation:** dynamic budget allocator and early-stop rules.
- **Risk:** Misleading conclusions from noisy metrics.
  - **Mitigation:** repeated runs, confidence intervals, robust reporting.

---

## 14) Immediate next build tasks (first coding sprint)

1. Scaffold monorepo structure and dev containers.
2. Define shared JSON schemas for agent task payloads.
3. Implement intake API and persistence for project/run entities.
4. Create web intake form and run timeline skeleton.
5. Implement orchestrator state machine with mock specialist agents.
6. Add notebook entry writer and citation model.
7. Add CI checks (lint/test/typecheck).

