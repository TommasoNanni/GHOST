#!/bin/bash -l
#SBATCH --job-name=test_depth_place
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus-per-node=0
#SBATCH --account=a144
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tnanni@ethz.ch

set -euo pipefail
cd /users/tnanni/ghost
ulimit -c 0

echo "Job ID: $SLURM_JOB_ID  |  Start: $(date)  |  Node: $SLURMD_NODENAME"

pixi run python scripts/test_depth_placement.py

echo "Done: $(date)"
