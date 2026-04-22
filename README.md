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

## Dataset
- This repository does **not** include UrbanSound8K on GitHub.
- Keep the dataset local and pass it with `--dataset-root /path/to/UrbanSound8K`.
- `.gitignore` excludes `UrbanSound8K/` and `results/` so the dataset and experiment artifacts are not committed accidentally.
- The code expects the standard UrbanSound8K layout:
  - `UrbanSound8K/audio/fold1 ... fold10`
  - `UrbanSound8K/metadata/UrbanSound8K.csv`

## Current experimental structure
- Edge model: compact CNN over log-mel spectrograms
- Confidence handling: temperature-scaled probabilities plus top-2 margin
- Middleware policy: class-priority-aware adaptive triage
- Priority support: per-class importance and false-positive cost maps in [`configs/default.yaml`](/Users/divijgupta/Desktop/SirenGate/configs/default.yaml)
- Evaluation outputs: per-event CSV traces and a `policy_summary.csv` for report plots

## Tested Results
The strongest tested edge checkpoint in this workspace is [`results/train_run`](/Users/divijgupta/Desktop/SirenGate/results/train_run), trained for 8 epochs on UrbanSound8K fold splits with fold 10 held out for validation.

Training result:
- Best validation accuracy: `0.5938`

Middleware evaluation using the corrected adaptive controller:
- Checkpoint: [`results/train_run/best_model.pt`](/Users/divijgupta/Desktop/SirenGate/results/train_run/best_model.pt)
- Simulation output: [`results/train_run_sim_default_fixed/policy_summary.csv`](/Users/divijgupta/Desktop/SirenGate/results/train_run_sim_default_fixed/policy_summary.csv)

Summary table:

| Policy | Accuracy | Macro F1 | Avg bytes/clip | Bandwidth reduction |
|---|---:|---:|---:|---:|
| `always_cloud` | 0.9008 | 0.8941 | 128000 | 0.0% |
| `edge_only` | 0.5938 | 0.5712 | 64 | 99.95% |
| `fixed_threshold` | 0.8805 | 0.8691 | 95901 | 25.1% |
| `adaptive_triage` | 0.8088 | 0.7912 | 53747 | 58.0% |

What this means:
- `edge_only` is cheap but not accurate enough to be the final system on its own.
- `always_cloud` is strongest on raw detection quality but spends the full bandwidth budget.
- `adaptive_triage` is the most useful project result so far: it cuts bandwidth by about `58%` versus `always_cloud` while improving macro F1 from `0.571` to `0.791` over `edge_only`.
- The corrected adaptive policy now uses all three routes on the validation stream:
  - `alert_only`: 379 clips
  - `embedding`: 107 clips
  - `raw_audio`: 351 clips

Priority-class behavior under `adaptive_triage`:
- `siren` F1 improves from `0.450` in `edge_only` to `0.827`.
- `car_horn` F1 improves from `0.493` in `edge_only` to `0.727`.
- `gun_shot` recall stays at `1.000`, but precision is still low (`0.421`), which means the edge model still confuses that class often and needs more training or better features.

Plots from the tested run:
- [`results/train_run_sim_default_fixed/bandwidth_vs_f1.png`](/Users/divijgupta/Desktop/SirenGate/results/train_run_sim_default_fixed/bandwidth_vs_f1.png)
- [`results/train_run_sim_default_fixed/adaptive_threshold_trace.png`](/Users/divijgupta/Desktop/SirenGate/results/train_run_sim_default_fixed/adaptive_threshold_trace.png)

## Notes and limitations
- The adaptive routing logic had a threshold-update sign bug earlier. The current results above are from the corrected controller in [`sirengate/middleware.py`](/Users/divijgupta/Desktop/SirenGate/sirengate/middleware.py).
- In the current local environment, 9 UrbanSound8K clips use WAV encodings that SciPy cannot decode. The loader skips those files automatically, so training runs on 8,723 clips instead of all 8,732.
- `fixed_triage` still behaves mostly like a two-route system with the current confidence distribution. The adaptive policy is currently the path that meaningfully exercises `alert / embedding / raw_audio`.
- The edge model is still the main bottleneck. If you want stronger final report numbers, the next practical step is more edge-model training or a better feature extractor, not another middleware rewrite.

## Reproducing the tested run
Train the stronger edge model:

```bash
python -m sirengate.cli train --dataset-root /path/to/UrbanSound8K --output-dir results/train_run
```

Run the simulation with the corrected adaptive middleware:

```bash
python -m sirengate.cli simulate --config configs/default.yaml --dataset-root /path/to/UrbanSound8K --checkpoint results/train_run/best_model.pt --output-dir results/train_run_sim_default_fixed
```
