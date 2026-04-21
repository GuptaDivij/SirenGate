#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m sirengate.cli demo --output-dir results/demo