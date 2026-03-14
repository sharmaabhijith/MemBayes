#!/bin/bash
#SBATCH --job-name=membayesv2
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%j_membayesv2-o.txt
#SBATCH --error=slurm_logs/%j_membayesv2-e.txt

mkdir -p slurm_logs

cd "${SLURM_SUBMIT_DIR}"

# Load API keys from .env
set -a
source .env
set +a

export PATH="/apps/local/anaconda3/bin:$PATH"

# Step 1: Generate benchmark (skips if already exists)
if [ ! -f results/benchmark.json ]; then
    echo "Generating benchmark..."
    python -u -m benchmark.generator --output results/benchmark.json
fi

# Step 2: Run experiments
python -u -m benchmark.run_experiments
