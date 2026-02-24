"""Experiment runner with deterministic simulation backends for local validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .simulator import ImageGenConfig, evaluate_imagegen_config


@dataclass
class RunnerJob:
    experiment_id: str
    config_path: str


def run_job(job: RunnerJob) -> dict[str, Any]:
    config_file = Path(job.config_path)
    if not config_file.exists():
        return {
            "experiment_id": job.experiment_id,
            "status": "failed",
            "message": f"Config file not found at {job.config_path}",
        }

    payload = json.loads(config_file.read_text())
    if payload.get("task") != "imagegen_eval":
        return {
            "experiment_id": job.experiment_id,
            "status": "failed",
            "message": "Unsupported task type; expected task='imagegen_eval'",
        }

    result = evaluate_imagegen_config(
        ImageGenConfig(
            dataset=payload["dataset"],
            noise_schedule=payload["noise_schedule"],
            sampler=payload["sampler"],
            guidance_scale=payload["guidance_scale"],
            learning_rate=payload["learning_rate"],
            ema_decay=payload["ema_decay"],
            grad_clip=payload["grad_clip"],
        )
    )

    return {
        "experiment_id": job.experiment_id,
        "status": "completed",
        "message": f"Simulated evaluation complete for {job.experiment_id}",
        "result": result.as_dict(),
    }
