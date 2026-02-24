from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import RunnerJob, run_job
from .simulator import run_imagenet_long_horizon_campaign


def _write_output(output_path: str | None, payload: dict) -> None:
    rendered = json.dumps(payload, indent=2)
    if output_path:
        Path(output_path).write_text(rendered + "\n")
    else:
        print(rendered)


def main() -> int:
    parser = argparse.ArgumentParser(description="ResearchSwarm deterministic runner CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_job_cmd = subparsers.add_parser("run-job", help="Run a single simulated runner job")
    run_job_cmd.add_argument("--experiment-id", required=True)
    run_job_cmd.add_argument("--config-path", required=True)
    run_job_cmd.add_argument("--output", required=False)

    campaign_cmd = subparsers.add_parser(
        "imagenet-campaign", help="Run long-horizon ImageNet-subset research simulation"
    )
    campaign_cmd.add_argument("--objective", default="Find robust high-quality ImageNet-subset imagegen settings")
    campaign_cmd.add_argument("--budget-hours", type=float, default=28.0)
    campaign_cmd.add_argument("--max-runs", type=int, default=12)
    campaign_cmd.add_argument("--output", required=False)

    args = parser.parse_args()

    if args.command == "run-job":
        payload = run_job(RunnerJob(experiment_id=args.experiment_id, config_path=args.config_path))
        _write_output(args.output, payload)
        return 0 if payload["status"] == "completed" else 2

    if args.command == "imagenet-campaign":
        report = run_imagenet_long_horizon_campaign(
            objective=args.objective,
            budget_hours=args.budget_hours,
            max_runs=args.max_runs,
        )
        _write_output(args.output, report.as_dict())
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
