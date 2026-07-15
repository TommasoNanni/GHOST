#!/bin/bash -l
#SBATCH --job-name=ma_baseline_eh
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --account=a144
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tnanni@ethz.ch
set -euo pipefail
cd /users/tnanni/ghost
ulimit -c 0
pixi run python scripts/run_ma_baseline_egohumans.py ${ACTIVITY:+--activity "$ACTIVITY"}
echo "Done: $(date)"
