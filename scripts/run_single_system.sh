#!/bin/bash
#SBATCH --job-name=membayes-%j
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_logs/%j_%x-o.txt
#SBATCH --error=slurm_logs/%j_%x-e.txt

# Usage: sbatch --job-name=membayes-vcl scripts/run_single_system.sh vcl
# The system name is passed as the first argument.

SYSTEM=${1:?"Usage: sbatch scripts/run_single_system.sh <system_name>"}

mkdir -p slurm_logs

cd "${SLURM_SUBMIT_DIR}"

set -a
source .env
set +a

export PATH="/apps/local/anaconda3/bin:$PATH"

echo "Running system: ${SYSTEM}"
python -u -m benchmark.run_experiments --systems ${SYSTEM}
