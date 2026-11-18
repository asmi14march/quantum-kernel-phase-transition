#!/usr/bin/env bash
# Reproduce every table and figure in results/.  Pass --quick for a ~1 minute smoke run.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore
PY=${PYTHON:-python3}
$PY scripts/generate_data.py "$@"
$PY scripts/run_benchmark.py "$@"
$PY scripts/run_critical_point.py "$@"
$PY scripts/run_scaling.py "$@"
$PY scripts/run_learning_and_noise.py "$@"
