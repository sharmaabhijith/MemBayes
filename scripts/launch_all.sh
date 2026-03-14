#!/bin/bash
# Launch all 6 systems as parallel SLURM jobs.
# Each job runs independently and saves results to results/<system_name>.json
#
# Usage: bash scripts/launch_all.sh

set -e
cd "$(dirname "$0")/.."

mkdir -p slurm_logs results

# Generate benchmark first (if not already present)
if [ ! -f results/benchmark.json ]; then
    echo "Generating benchmark..."
    export PATH="/apps/local/anaconda3/bin:$PATH"
    set -a; source .env; set +a
    python -u -m benchmark.generator --output results/benchmark.json
    echo "Done."
fi

echo ""
echo "Submitting 6 parallel jobs..."
echo "==============================="

SYSTEMS=("vcl" "vcl_no_coreset" "vcl_no_decay" "naive" "sliding_window" "decay_only")

for sys in "${SYSTEMS[@]}"; do
    JOB_ID=$(sbatch --job-name="mb-${sys}" --parsable scripts/run_single_system.sh "${sys}")
    echo "  ${sys} -> Job ${JOB_ID}"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u $USER"
echo "Results will be in: results/<system_name>_results.json"
echo ""
echo "After all jobs complete, run:"
echo "  python -m benchmark.run_experiments --merge-only"
