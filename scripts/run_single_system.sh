#!/bin/bash
# Run a single system experiment.
# Usage: bash scripts/run_single_system.sh <system_name> [--run-dir <dir>]

SYSTEM=${1:?"Usage: bash scripts/run_single_system.sh <system_name> [--run-dir <dir>]"}
shift

cd "$(dirname "$0")/.."

set -a
source .env
set +a

echo "Running system: ${SYSTEM}"
python -u -m evaluation.runner --systems ${SYSTEM} "$@"
