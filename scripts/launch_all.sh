#!/bin/bash
# Launch systems as parallel background processes.
# Each process writes to the same timestamped run directory.
#
# Usage: bash scripts/launch_all.sh

set -e
cd "$(dirname "$0")/.."

mkdir -p results

# Load environment
set -a; source .env; set +a

# Generate benchmark (always regenerate to pick up generator changes)
echo "Generating expanded benchmark (H1-H14)..."
python -u -m evaluation.generator --output results/benchmark.json
echo "Done."

# Create a shared timestamped run directory with logs subfolder
RUN_DIR="results/run_$(date +%Y-%m-%d_%H-%M-%S)"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

echo ""
echo "Launching parallel processes..."
echo "==================================="
echo "  Run directory: ${RUN_DIR}"
echo ""

#SYSTEMS=("vcl" "vcl_no_coreset" "vcl_no_decay" "naive" "sliding_window" "decay_only")
SYSTEMS=("vcl_no_decay")



for sys in "${SYSTEMS[@]}"; do
    echo "  Starting ${sys}..."
    bash scripts/run_single_system.sh "${sys}" --run-dir "${RUN_DIR}" > "${LOG_DIR}/${sys}.log" 2>&1 &
    PIDS+=($!)
    echo "  ${sys} -> PID $!"
done

echo ""
echo "All processes launched. Waiting for completion..."
echo "Logs: ${LOG_DIR}/<system_name>.log"
echo ""

# Wait for all and track failures
FAILED=()
for i in "${!SYSTEMS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  [DONE] ${SYSTEMS[$i]}"
    else
        echo "  [FAIL] ${SYSTEMS[$i]} (see ${LOG_DIR}/${SYSTEMS[$i]}.log)"
        FAILED+=("${SYSTEMS[$i]}")
    fi
done

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All systems completed successfully."
    echo "Results saved to: ${RUN_DIR}/"
    echo ""
    echo "To merge results and generate plots, run:"
    echo "  python -m evaluation.runner --merge-only --run-dir ${RUN_DIR}"
else
    echo "Failed systems: ${FAILED[*]}"
    echo "Check logs/ for details."
    echo ""
    echo "To merge results from successful systems, run:"
    echo "  python -m evaluation.runner --merge-only --run-dir ${RUN_DIR}"
    exit 1
fi
