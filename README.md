# SirenGate

SirenGate is an adaptive Edge-AI middleware project for urban audio sensing.

## What it does
- trains a lightweight edge classifier on urban sound clips
- calibrates confidence with temperature scaling
- decides whether to send:
  - a tiny alert only, or
  - full raw audio to the cloud
- compares bandwidth vs detection performance

## Repo structure
- `sirengate/data.py` - dataset loading and preprocessing
- `sirengate/models.py` - small CNN and training
- `sirengate/calibration.py` - temperature scaling
- `sirengate/middleware.py` - adaptive routing logic
- `sirengate/simulation.py` - end-to-end policy evaluation
- `sirengate/plots.py` - report figures
- `sirengate/cli.py` - command-line entrypoint

## Quick start

### Demo mode
```bash
python -m sirengate.cli demo --output-dir results/demo