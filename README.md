# SirenGate

SirenGate is an adaptive Edge-AI middleware project for urban audio sensing.

## What it does
- trains a lightweight edge classifier on urban sound clips
- calibrates confidence with temperature scaling
- simulates three transmission modes:
  - `alert_only` for high-confidence local detections
  - `embedding` for medium-confidence uncertain detections
  - `raw_audio` for low-confidence or high-risk detections
- adapts routing thresholds to balance bandwidth budgets against detection quality
- compares bandwidth vs detection performance, including priority-weighted metrics

## Repo structure
- `sirengate/data.py` - dataset loading and preprocessing
- `sirengate/models.py` - small CNN and training
- `sirengate/calibration.py` - temperature scaling
- `sirengate/middleware.py` - adaptive multi-route routing logic
- `sirengate/simulation.py` - end-to-end policy evaluation
- `sirengate/plots.py` - report figures
- `sirengate/cli.py` - command-line entrypoint

## Quick start

### Demo mode
```bash
python -m sirengate.cli demo --output-dir results/demo
```

### Train on UrbanSound8K
```bash
python -m sirengate.cli train --dataset-root /path/to/UrbanSound8K --output-dir results/train
```

### Train from a CSV manifest
```bash
python -m sirengate.cli train --manifest data/clips.csv --output-dir results/train
```

Manifest columns:
- required: `path`, `label`
- optional: `fold`, `clip_id`

If `path` is relative, it is resolved relative to the manifest file.

## Current experimental structure
- Edge model: compact CNN over log-mel spectrograms
- Confidence handling: temperature-scaled probabilities plus top-2 margin
- Middleware policy: class-priority-aware adaptive triage
- Priority support: per-class importance and false-positive cost maps in [`configs/default.yaml`](/Users/divijgupta/Desktop/SirenGate/configs/default.yaml)
- Evaluation outputs: per-event CSV traces and a `policy_summary.csv` for report plots

## Notes for your project scope
- The simulator now matches your updated idea better than the original two-path version because it supports `label / embedding / raw audio` routing.
- `gun_shot` and `siren` can be given stricter escalation behavior through `priority_weights` and `false_positive_costs`.
- The code supports a CSV ingest path now, so when your dataset manifest arrives you do not need to reorganize the project again.
