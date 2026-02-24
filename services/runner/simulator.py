"""Deterministic simulation utilities for image-generation research workflows.

These helpers intentionally avoid heavyweight ML dependencies so orchestrator/runner
integration can be tested end-to-end in CI and local development.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


Dataset = Literal["mnist", "imagenet_subset"]
NoiseSchedule = Literal["linear", "cosine", "sigmoid"]
Sampler = Literal["ddpm", "ddim", "heun"]


@dataclass(frozen=True)
class ImageGenConfig:
    dataset: Dataset
    noise_schedule: NoiseSchedule
    sampler: Sampler
    guidance_scale: float
    learning_rate: float
    ema_decay: float
    grad_clip: float


@dataclass(frozen=True)
class ExperimentResult:
    config: ImageGenConfig
    fid_proxy: float
    stability_score: float
    throughput: float
    memory_gb: float
    compute_hours: float
    score: float

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["config"] = asdict(self.config)
        return payload


@dataclass(frozen=True)
class CampaignReport:
    objective: str
    dataset: Dataset
    budget_hours: float
    consumed_hours: float
    completed_runs: int
    stopped_reason: str
    best_result: dict
    phase_summaries: list[dict]

    def as_dict(self) -> dict:
        return asdict(self)


def _validate_config(config: ImageGenConfig) -> None:
    if not (0.1 <= config.guidance_scale <= 12.0):
        raise ValueError("guidance_scale must be in [0.1, 12.0]")
    if not (1e-5 <= config.learning_rate <= 5e-3):
        raise ValueError("learning_rate must be in [1e-5, 5e-3]")
    if not (0.9 <= config.ema_decay <= 0.9999):
        raise ValueError("ema_decay must be in [0.9, 0.9999]")
    if not (0.1 <= config.grad_clip <= 5.0):
        raise ValueError("grad_clip must be in [0.1, 5.0]")


def evaluate_imagegen_config(config: ImageGenConfig) -> ExperimentResult:
    """Evaluate a config with deterministic proxy metrics.

    Proxy behavior differs by dataset to emulate longer horizon ImageNet exploration.
    """

    _validate_config(config)

    dataset_base = {"mnist": 22.5, "imagenet_subset": 46.0}[config.dataset]
    dataset_compute = {"mnist": 0.2, "imagenet_subset": 2.5}[config.dataset]
    dataset_memory = {"mnist": 1.2, "imagenet_subset": 3.8}[config.dataset]

    schedule_bonus = {"linear": 0.0, "cosine": 2.3, "sigmoid": 1.5}[config.noise_schedule]
    sampler_bonus = {"ddpm": 0.0, "ddim": 1.8, "heun": 2.6}[config.sampler]

    lr_penalty = abs(config.learning_rate - 0.0008) * (9000 if config.dataset == "mnist" else 12000)
    guidance_penalty = abs(config.guidance_scale - 4.5) * 0.95
    ema_bonus = (config.ema_decay - 0.97) * 75
    clip_penalty = abs(config.grad_clip - 1.0) * 1.2

    fid_proxy = dataset_base - schedule_bonus - sampler_bonus + lr_penalty + guidance_penalty - ema_bonus + clip_penalty
    stability_score = max(0.0, min(1.0, 0.65 + ema_bonus / 28 - lr_penalty / 28 - clip_penalty / 9))
    throughput = max(4.0, (140.0 if config.dataset == "mnist" else 36.0) - config.guidance_scale * 2.1)
    memory_gb = dataset_memory + config.guidance_scale * 0.2 + (2.1 if config.sampler == "heun" else 1.1)
    compute_hours = max(0.05, dataset_compute + config.guidance_scale * 0.15 + (0.7 if config.sampler == "heun" else 0.25))

    score = (58.0 - fid_proxy) + stability_score * 12 + throughput / 28 - memory_gb / 5 - compute_hours / 8

    return ExperimentResult(
        config=config,
        fid_proxy=round(fid_proxy, 4),
        stability_score=round(stability_score, 4),
        throughput=round(throughput, 3),
        memory_gb=round(memory_gb, 3),
        compute_hours=round(compute_hours, 3),
        score=round(score, 4),
    )


def _sort_results(results: list[ExperimentResult]) -> list[ExperimentResult]:
    if not results:
        raise ValueError("No results to rank")
    return sorted(results, key=lambda r: (r.score, -r.fid_proxy, r.compute_hours), reverse=True)


def run_imagenet_long_horizon_campaign(
    objective: str,
    budget_hours: float,
    max_runs: int = 12,
) -> CampaignReport:
    """Simulate a phased ImageNet-subset campaign with budget-aware stopping."""

    if budget_hours <= 0:
        raise ValueError("budget_hours must be positive")
    if max_runs <= 0:
        raise ValueError("max_runs must be positive")

    phase_candidates: list[tuple[str, list[ImageGenConfig]]] = [
        (
            "phase_a_exploration",
            [
                ImageGenConfig("imagenet_subset", "linear", "ddpm", 3.0, 0.0010, 0.97, 1.0),
                ImageGenConfig("imagenet_subset", "cosine", "ddim", 4.0, 0.0009, 0.985, 1.0),
                ImageGenConfig("imagenet_subset", "sigmoid", "heun", 4.5, 0.0008, 0.99, 1.0),
            ],
        ),
        (
            "phase_b_refinement",
            [
                ImageGenConfig("imagenet_subset", "cosine", "heun", 4.5, 0.0008, 0.991, 1.0),
                ImageGenConfig("imagenet_subset", "cosine", "ddim", 5.0, 0.00075, 0.992, 0.9),
                ImageGenConfig("imagenet_subset", "sigmoid", "heun", 4.2, 0.00078, 0.993, 0.9),
            ],
        ),
        (
            "phase_c_stability",
            [
                ImageGenConfig("imagenet_subset", "cosine", "heun", 4.6, 0.0008, 0.994, 0.85),
                ImageGenConfig("imagenet_subset", "cosine", "ddim", 4.8, 0.00082, 0.995, 0.8),
                ImageGenConfig("imagenet_subset", "sigmoid", "heun", 4.4, 0.00079, 0.995, 0.85),
            ],
        ),
    ]

    consumed = 0.0
    completed_runs = 0
    all_results: list[ExperimentResult] = []
    phase_summaries: list[dict] = []
    stopped_reason = "completed_all_phases"

    for phase_name, candidates in phase_candidates:
        phase_results: list[ExperimentResult] = []
        for candidate in candidates:
            if completed_runs >= max_runs:
                stopped_reason = "max_runs_reached"
                break
            result = evaluate_imagegen_config(candidate)
            if consumed + result.compute_hours > budget_hours:
                stopped_reason = "budget_exhausted"
                break
            consumed += result.compute_hours
            completed_runs += 1
            all_results.append(result)
            phase_results.append(result)

        if phase_results:
            phase_best = _sort_results(phase_results)[0]
            phase_summaries.append(
                {
                    "phase": phase_name,
                    "runs": len(phase_results),
                    "best_score": phase_best.score,
                    "best_fid_proxy": phase_best.fid_proxy,
                }
            )

        if stopped_reason != "completed_all_phases":
            break

    if not all_results:
        raise ValueError("No campaign runs could be executed within budget")

    best = _sort_results(all_results)[0]

    return CampaignReport(
        objective=objective,
        dataset="imagenet_subset",
        budget_hours=budget_hours,
        consumed_hours=round(consumed, 3),
        completed_runs=completed_runs,
        stopped_reason=stopped_reason,
        best_result=best.as_dict(),
        phase_summaries=phase_summaries,
    )
