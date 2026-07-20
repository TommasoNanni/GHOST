#!/bin/bash -l
#SBATCH --job-name=ma_baseline_egoexo
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
# Resumable: takes with mapanything_scale_baseline.npy are skipped, so relaunch
# until SWEEP_DONE reports failed=0 (debug partition caps at 1:30).
pixi run python scripts/run_ma_baseline_egoexo.py ${TAKE:+--take "$TAKE"}
echo "Done: $(date)"
