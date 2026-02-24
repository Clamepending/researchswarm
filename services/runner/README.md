# Runner Service

Container entrypoint for training/evaluation execution.

## Current status

Runner includes a deterministic simulation backend for local and CI validation of image-generation
research loops without GPU training dependencies.

## CLI usage

Run one simulated experiment from JSON config:

```bash
python -m services.runner.cli run-job --experiment-id exp-001 --config-path /path/to/config.json
```

Run a long-horizon ImageNet-subset campaign with budget-aware stopping:

```bash
python -m services.runner.cli imagenet-campaign --budget-hours 28 --max-runs 12
```

### Config format (`run-job`)

```json
{
  "task": "imagegen_eval",
  "dataset": "mnist",
  "noise_schedule": "cosine",
  "sampler": "heun",
  "guidance_scale": 4.5,
  "learning_rate": 0.0008,
  "ema_decay": 0.99,
  "grad_clip": 1.0
}
```


### Tiny synthetic ImageNet training sweep

Use this to compare diffusion-style noise schedules and prediction targets with a very small model:

```bash
python -m services.runner.cli tiny-imagenet-train --learning-rate 0.01 --hidden-size 8 --train-steps 50
```
