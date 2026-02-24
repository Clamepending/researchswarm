"""Experiment runner skeleton for Phase 1 scaffolding."""

from dataclasses import dataclass


@dataclass
class RunnerJob:
    experiment_id: str
    config_path: str


def run_job(job: RunnerJob) -> dict[str, str]:
    return {
        "experiment_id": job.experiment_id,
        "status": "queued",
        "message": f"Runner stub accepted config at {job.config_path}",
    }
