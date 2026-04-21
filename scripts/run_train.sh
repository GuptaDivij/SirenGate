#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/UrbanSound8K"
  exit 1
fi

cd "$(dirname "$0")/.."
python -m sirengate.cli train --dataset-root "$1" --output-dir results/train_run
python -m sirengate.cli simulate --dataset-root "$1" --checkpoint results/train_run/best_model.pt --output-dir results/sim_run