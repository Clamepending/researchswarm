from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


NoiseSchedule = Literal["linear", "cosine", "sigmoid"]
Sampler = Literal["ddpm", "ddim", "heun"]


@dataclass(frozen=True)
class MnistImageGenConfig:
    noise_schedule: NoiseSchedule
    sampler: Sampler
    guidance_scale: float
    learning_rate: float
    ema_decay: float
    grad_clip: float


@dataclass(frozen=True)
class MnistImageGenResult:
    config: MnistImageGenConfig
    fid_proxy: float
    stability_score: float
    throughput: float
    memory_gb: float
    score: float


def _validate_config(config: MnistImageGenConfig) -> None:
    if not (0.1 <= config.guidance_scale <= 12.0):
        raise ValueError("guidance_scale must be in [0.1, 12.0]")
    if not (1e-5 <= config.learning_rate <= 5e-3):
        raise ValueError("learning_rate must be in [1e-5, 5e-3]")
    if not (0.9 <= config.ema_decay <= 0.9999):
        raise ValueError("ema_decay must be in [0.9, 0.9999]")
    if not (0.1 <= config.grad_clip <= 5.0):
        raise ValueError("grad_clip must be in [0.1, 5.0]")


def evaluate_config(config: MnistImageGenConfig) -> MnistImageGenResult:
    """Deterministic proxy objective for fast local experimentation.

    Lower fid_proxy is better. Higher stability/throughput is better.
    """

    _validate_config(config)

    # Baseline values roughly tuned to make plausible trade-offs.
    schedule_bonus = {"linear": 0.0, "cosine": 2.2, "sigmoid": 1.4}[config.noise_schedule]
    sampler_bonus = {"ddpm": 0.0, "ddim": 1.8, "heun": 2.4}[config.sampler]

    lr_penalty = abs(config.learning_rate - 0.0008) * 9000
    guidance_penalty = abs(config.guidance_scale - 4.5) * 0.95
    ema_bonus = (config.ema_decay - 0.97) * 80
    clip_penalty = abs(config.grad_clip - 1.0) * 1.2

    fid_proxy = 22.5 - schedule_bonus - sampler_bonus + lr_penalty + guidance_penalty - ema_bonus + clip_penalty
    stability_score = max(0.0, min(1.0, 0.66 + ema_bonus / 30 - lr_penalty / 25 - clip_penalty / 10))
    throughput = max(10.0, 150.0 - config.guidance_scale * 11 - (0.9999 - config.ema_decay) * 500)
    memory_gb = 1.7 + config.guidance_scale * 0.18 + (2.2 if config.sampler == "heun" else 0.9)

    score = (40.0 - fid_proxy) + stability_score * 10 + throughput / 45 - memory_gb / 6

    return MnistImageGenResult(
        config=config,
        fid_proxy=round(fid_proxy, 4),
        stability_score=round(stability_score, 4),
        throughput=round(throughput, 3),
        memory_gb=round(memory_gb, 3),
        score=round(score, 4),
    )


def rank_results(results: list[MnistImageGenResult]) -> list[MnistImageGenResult]:
    if not results:
        raise ValueError("No experiment results to rank")

    # Score first; for ties, prefer lower FID and then higher throughput.
    return sorted(results, key=lambda r: (r.score, -r.fid_proxy, r.throughput), reverse=True)
