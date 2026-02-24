import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.runner.runner import RunnerJob, run_job
from services.runner.simulator import ImageGenConfig, evaluate_imagegen_config, run_imagenet_long_horizon_campaign


def test_evaluate_imagegen_config_mnist_smoke() -> None:
    result = evaluate_imagegen_config(
        ImageGenConfig(
            dataset="mnist",
            noise_schedule="cosine",
            sampler="heun",
            guidance_scale=4.5,
            learning_rate=0.0008,
            ema_decay=0.99,
            grad_clip=1.0,
        )
    )
    assert result.fid_proxy > 0
    assert 0 <= result.stability_score <= 1
    assert result.compute_hours > 0


def test_long_horizon_campaign_respects_budget_and_returns_phases() -> None:
    report = run_imagenet_long_horizon_campaign(
        objective="Long horizon imagenet test case",
        budget_hours=20.0,
        max_runs=8,
    )
    assert report.dataset == "imagenet_subset"
    assert report.consumed_hours <= report.budget_hours
    assert report.completed_runs > 0
    assert len(report.phase_summaries) >= 1


def test_runner_job_with_missing_file_fails() -> None:
    response = run_job(RunnerJob(experiment_id="missing", config_path="/tmp/not-here.json"))
    assert response["status"] == "failed"


def test_cli_imagenet_campaign_writes_output(tmp_path: Path) -> None:
    output_path = tmp_path / "campaign.json"
    cmd = [
        "python",
        "-m",
        "services.runner.cli",
        "imagenet-campaign",
        "--budget-hours",
        "14",
        "--max-runs",
        "5",
        "--output",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    payload = json.loads(output_path.read_text())
    assert payload["dataset"] == "imagenet_subset"
    assert payload["completed_runs"] > 0
