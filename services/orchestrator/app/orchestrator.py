from __future__ import annotations

from dataclasses import dataclass
import json
from uuid import UUID

from .mnist_imagegen import MnistImageGenConfig, evaluate_config, rank_results
from .models import (
    MnistImageGenBatchReport,
    MnistImageGenCandidate,
    MnistImageGenEvaluation,
    NotebookEntry,
    TimelineEvent,
)
from .storage import add_experiment_run, add_notebook_entry, add_timeline_event, update_run_state


@dataclass
class SpecialistOutput:
    agent: str
    summary: str
    confidence: float


class MockSpecialist:
    def __init__(self, name: str, template: str) -> None:
        self.name = name
        self.template = template

    def execute(self, objective: str) -> SpecialistOutput:
        return SpecialistOutput(
            agent=self.name,
            summary=self.template.format(objective=objective),
            confidence=0.68,
        )


class OrchestratorStateMachine:
    def __init__(self) -> None:
        self.specialists = [
            MockSpecialist("literature", "Collected relevant priors for '{objective}'."),
            MockSpecialist("experiment_designer", "Drafted an experiment matrix targeting '{objective}'."),
            MockSpecialist("evaluator", "Prepared evaluation rubric and significance checks for '{objective}'."),
        ]

    def run_discovery(self, run_id: UUID, objective: str) -> list[SpecialistOutput]:
        add_timeline_event(
            TimelineEvent(run_id=run_id, event_type="stage_transition", message="Entering discovery phase", confidence=0.9)
        )

        outputs: list[SpecialistOutput] = []
        for specialist in self.specialists:
            output = specialist.execute(objective)
            outputs.append(output)
            add_timeline_event(
                TimelineEvent(
                    run_id=run_id,
                    event_type="agent_note",
                    message=f"[{output.agent}] {output.summary}",
                    confidence=output.confidence,
                )
            )
            add_notebook_entry(
                NotebookEntry(
                    run_id=run_id,
                    author_agent=output.agent,
                    summary=output.summary,
                    decision="Continue to next specialist",
                    citations=[],
                )
            )
        return outputs

    def run_mnist_imagegen_batch(
        self,
        run_id: UUID,
        objective: str,
        candidates: list[MnistImageGenCandidate],
    ) -> MnistImageGenBatchReport:
        if len(candidates) == 0:
            raise ValueError("At least one candidate config is required")

        update_run_state(run_id, status="running", stage="planning")
        add_timeline_event(
            TimelineEvent(
                run_id=run_id,
                event_type="stage_transition",
                message="Planning MNIST image-generation config search",
                confidence=0.93,
            )
        )

        seen: set[tuple] = set()
        unique_candidates: list[MnistImageGenCandidate] = []
        for candidate in candidates:
            key = (
                candidate.noise_schedule,
                candidate.sampler,
                round(candidate.guidance_scale, 6),
                round(candidate.learning_rate, 8),
                round(candidate.ema_decay, 6),
                round(candidate.grad_clip, 6),
            )
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append(candidate)

        rejected = len(candidates) - len(unique_candidates)

        update_run_state(run_id, stage="execution")
        add_timeline_event(
            TimelineEvent(
                run_id=run_id,
                event_type="stage_transition",
                message=f"Executing {len(unique_candidates)} candidate experiments",
                confidence=0.9,
            )
        )

        raw_results = []
        for candidate in unique_candidates:
            config = MnistImageGenConfig(**candidate.model_dump())
            result = evaluate_config(config)
            raw_results.append(result)
            add_experiment_run(
                run_id=run_id,
                experiment_family="mnist_image_generation",
                config_json=candidate.model_dump_json(),
                metrics_json=json.dumps({
                    "fid_proxy": result.fid_proxy,
                    "stability_score": result.stability_score,
                    "throughput": result.throughput,
                    "memory_gb": result.memory_gb,
                    "score": result.score,
                }),
            )

        if not raw_results:
            update_run_state(run_id, status="failed", stage="evaluation")
            add_timeline_event(
                TimelineEvent(run_id=run_id, event_type="failure", message="No valid candidates survived deduplication", confidence=1.0)
            )
            raise ValueError("No valid candidates to evaluate")

        ranked = rank_results(raw_results)
        evaluations: list[MnistImageGenEvaluation] = []
        for idx, result in enumerate(ranked, start=1):
            evaluations.append(
                MnistImageGenEvaluation(
                    rank=idx,
                    noise_schedule=result.config.noise_schedule,
                    sampler=result.config.sampler,
                    guidance_scale=result.config.guidance_scale,
                    learning_rate=result.config.learning_rate,
                    ema_decay=result.config.ema_decay,
                    grad_clip=result.config.grad_clip,
                    fid_proxy=result.fid_proxy,
                    stability_score=result.stability_score,
                    throughput=result.throughput,
                    memory_gb=result.memory_gb,
                    score=result.score,
                )
            )

        best = evaluations[0]
        caveats = []
        if rejected > 0:
            caveats.append(f"{rejected} duplicate candidate(s) were skipped")
        if best.stability_score < 0.7:
            caveats.append("Best config has moderate stability; consider repeated-seed runs")
        if best.memory_gb > 4.0:
            caveats.append("Best config may exceed low-memory GPUs")

        add_notebook_entry(
            NotebookEntry(
                run_id=run_id,
                author_agent="evaluation",
                summary=(
                    f"Best setting: schedule={best.noise_schedule}, sampler={best.sampler}, "
                    f"FID proxy={best.fid_proxy}, stability={best.stability_score}."
                ),
                decision="Promote best configuration to follow-up validation.",
                citations=[],
            )
        )

        update_run_state(run_id, status="completed", stage="reporting")
        add_timeline_event(
            TimelineEvent(
                run_id=run_id,
                event_type="agent_note",
                message=f"MNIST image-generation batch completed. Best score={best.score}.",
                confidence=0.91,
            )
        )

        return MnistImageGenBatchReport(
            run_id=run_id,
            objective=objective,
            total_candidates=len(candidates),
            evaluated_candidates=len(unique_candidates),
            rejected_candidates=rejected,
            best=best,
            rankings=evaluations,
            caveats=caveats,
        )
