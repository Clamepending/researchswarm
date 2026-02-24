"""Tiny synthetic ImageNet-style denoising benchmark.

This module emulates diffusion training choices (noise schedule + prediction target)
with a tiny MLP trained on a fixed synthetic dataset so we can compare candidates
in CI without GPUs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Literal


NoiseSchedule = Literal["linear", "cosine", "sigmoid"]
PredictionTarget = Literal["epsilon", "v_prediction"]


@dataclass(frozen=True)
class TinyImagenetConfig:
    noise_schedule: NoiseSchedule
    prediction_target: PredictionTarget
    learning_rate: float
    hidden_size: int
    train_steps: int


@dataclass(frozen=True)
class TinyImagenetResult:
    config: TinyImagenetConfig
    train_loss: float
    val_loss: float
    stability_score: float
    throughput: float
    score: float

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["config"] = asdict(self.config)
        return payload


def _noise_curve(schedule: NoiseSchedule, steps: int = 32) -> list[float]:
    curve: list[float] = []
    for i in range(steps):
        x = i / max(1, steps - 1)
        if schedule == "linear":
            beta = 0.01 + 0.18 * x
        elif schedule == "cosine":
            beta = 0.008 + 0.17 * (1 - math.cos(math.pi * x)) / 2
        else:
            beta = 0.01 + 0.17 / (1 + math.exp(-8 * (x - 0.5)))
        curve.append(min(0.95, max(0.001, beta)))
    return curve


def _build_dataset(seed: int, samples: int, dims: int) -> list[tuple[list[float], list[float], int]]:
    rng = random.Random(seed)
    rows: list[tuple[list[float], list[float], int]] = []
    for _ in range(samples):
        x0 = [rng.uniform(-1.0, 1.0) for _ in range(dims)]
        eps = [rng.gauss(0.0, 1.0) for _ in range(dims)]
        t = rng.randrange(0, 32)
        rows.append((x0, eps, t))
    return rows


def _target(target_type: PredictionTarget, x0: list[float], eps: list[float], alpha: float) -> list[float]:
    if target_type == "epsilon":
        return eps
    s1 = math.sqrt(alpha)
    s2 = math.sqrt(max(0.0, 1.0 - alpha))
    return [s1 * e - s2 * x for x, e in zip(x0, eps)]


def run_tiny_imagenet_training(config: TinyImagenetConfig) -> TinyImagenetResult:
    if not (1e-4 <= config.learning_rate <= 0.03):
        raise ValueError("learning_rate must be in [1e-4, 0.03]")
    if not (4 <= config.hidden_size <= 64):
        raise ValueError("hidden_size must be in [4, 64]")
    if not (10 <= config.train_steps <= 160):
        raise ValueError("train_steps must be in [10, 160]")

    dims = 16
    in_dim = dims + 1
    schedule = _noise_curve(config.noise_schedule)
    train_data = _build_dataset(seed=7, samples=96, dims=dims)
    val_data = _build_dataset(seed=11, samples=40, dims=dims)

    init_rng = random.Random(23)
    w1 = [[init_rng.uniform(-0.2, 0.2) for _ in range(in_dim)] for _ in range(config.hidden_size)]
    b1 = [0.0 for _ in range(config.hidden_size)]
    w2 = [[init_rng.uniform(-0.2, 0.2) for _ in range(config.hidden_size)] for _ in range(dims)]
    b2 = [0.0 for _ in range(dims)]

    losses: list[float] = []
    lr = config.learning_rate

    for _ in range(config.train_steps):
        grad_w1 = [[0.0 for _ in range(in_dim)] for _ in range(config.hidden_size)]
        grad_b1 = [0.0 for _ in range(config.hidden_size)]
        grad_w2 = [[0.0 for _ in range(config.hidden_size)] for _ in range(dims)]
        grad_b2 = [0.0 for _ in range(dims)]
        total_loss = 0.0

        for x0, eps, t in train_data:
            alpha = 1.0 - schedule[t]
            s1 = math.sqrt(alpha)
            s2 = math.sqrt(max(0.0, 1.0 - alpha))
            xt = [s1 * clean + s2 * noise for clean, noise in zip(x0, eps)]
            model_in = xt + [t / 31.0]
            target = _target(config.prediction_target, x0, eps, alpha)

            hidden = []
            for h in range(config.hidden_size):
                z = b1[h]
                for j in range(in_dim):
                    z += w1[h][j] * model_in[j]
                hidden.append(math.tanh(z))

            output = []
            for o in range(dims):
                y = b2[o]
                for h in range(config.hidden_size):
                    y += w2[o][h] * hidden[h]
                output.append(y)

            deltas_out = [0.0 for _ in range(dims)]
            sample_loss = 0.0
            for o in range(dims):
                err = output[o] - target[o]
                sample_loss += err * err
                deltas_out[o] = 2.0 * err / dims
                grad_b2[o] += deltas_out[o]
                for h in range(config.hidden_size):
                    grad_w2[o][h] += deltas_out[o] * hidden[h]

            deltas_h = [0.0 for _ in range(config.hidden_size)]
            for h in range(config.hidden_size):
                back = 0.0
                for o in range(dims):
                    back += deltas_out[o] * w2[o][h]
                deltas_h[h] = back * (1.0 - hidden[h] * hidden[h])
                grad_b1[h] += deltas_h[h]
                for j in range(in_dim):
                    grad_w1[h][j] += deltas_h[h] * model_in[j]

            total_loss += sample_loss / dims

        n = float(len(train_data))
        for h in range(config.hidden_size):
            b1[h] -= lr * grad_b1[h] / n
            for j in range(in_dim):
                w1[h][j] -= lr * grad_w1[h][j] / n

        for o in range(dims):
            b2[o] -= lr * grad_b2[o] / n
            for h in range(config.hidden_size):
                w2[o][h] -= lr * grad_w2[o][h] / n

        losses.append(total_loss / n)

    def compute_loss(rows: list[tuple[list[float], list[float], int]]) -> float:
        total = 0.0
        for x0, eps, t in rows:
            alpha = 1.0 - schedule[t]
            s1 = math.sqrt(alpha)
            s2 = math.sqrt(max(0.0, 1.0 - alpha))
            xt = [s1 * clean + s2 * noise for clean, noise in zip(x0, eps)]
            model_in = xt + [t / 31.0]
            target = _target(config.prediction_target, x0, eps, alpha)

            hidden = []
            for h in range(config.hidden_size):
                z = b1[h]
                for j in range(in_dim):
                    z += w1[h][j] * model_in[j]
                hidden.append(math.tanh(z))

            sample_loss = 0.0
            for o in range(dims):
                y = b2[o]
                for h in range(config.hidden_size):
                    y += w2[o][h] * hidden[h]
                diff = y - target[o]
                sample_loss += diff * diff
            total += sample_loss / dims
        return total / len(rows)

    train_loss = losses[-1]
    val_loss = compute_loss(val_data)
    stability = max(0.0, min(1.0, 1.0 / (1.0 + val_loss * 3.5)))
    throughput = 520.0 / (config.hidden_size * 0.8 + config.train_steps * 0.3)
    score = (1.8 - val_loss) * 20 + stability * 8 + throughput / 12

    return TinyImagenetResult(
        config=config,
        train_loss=round(train_loss, 5),
        val_loss=round(val_loss, 5),
        stability_score=round(stability, 5),
        throughput=round(throughput, 3),
        score=round(score, 5),
    )


def rank_tiny_imagenet_results(results: list[TinyImagenetResult]) -> list[TinyImagenetResult]:
    if not results:
        raise ValueError("No results to rank")
    return sorted(results, key=lambda r: (r.score, -r.val_loss, r.throughput), reverse=True)

